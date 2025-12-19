import re
import math
import random
import itertools
import pandas as pd
from collections import Counter

pd.set_option('mode.chained_assignment', None)
from LUNAR.utils import verify_template_for_log_with_first_token
from LUNAR.utils import preprocess_log_for_query
from LUNAR.log_partition.text_distance import calculate_jaccard_one_to_many


class BaseClustering:

    def __init__(self,
                 sample_method="lcu_sampling",
                 sample_size=3,
                 min_cluster_size=100,
                 sample_min_similarity=0.5,
                 lcu_lamb=0.5,
                 lcu_sample_size=3,
                 sample_size_auto="fixed",
                 add_regex="add",
                 regex=[],
                 add_skip_sim=False,
                 pad_query=True):
        self.df_logs = None
        self.log_path = None
        self.num_total_logs = 0
        self.num_processed_logs = 0
        self.add_regex = add_regex
        self.regex = regex
        self.sample_method = sample_method
        self.min_cluster_size = min_cluster_size
        self.sample_min_similarity = sample_min_similarity
        self.lcu_lamb = lcu_lamb
        self.lcu_sample_size = lcu_sample_size
        self.sample_size = sample_size
        self.sample_size_assigned = sample_size
        self.max_sample_size, self.min_sample_size = 5, 1
        self.max_log_length, self.min_log_length = -1, -1
        if sample_size_auto == "auto":
            self.sample_size_auto = True
        else:
            self.sample_size_auto = False
        self.pad_query = pad_query
        self.add_skip_sim = add_skip_sim
        self.log_lengths = []
        self.clusters = {}
        self.update_map_parent2child = {}
        self.update_map_child2parent = {}

    def load_data(self, df_logs, log_path):
        print("Clustering load data")
        self.log_path = log_path
        self.df_logs = df_logs
        # debug
        print('self.add_regex: {}'.format(self.add_regex))
        print('self.regex: {}'.format(self.regex))
        if self.add_regex == "before":
            print("Clustering add regex before preprocess")
            self.df_logs.loc[:, "Content"] = self.df_logs.apply(
                lambda row: preprocess_log_for_query(row["Content"], self.regex
                                                     ),
                axis=1)

        self.df_logs = self.df_logs.assign(Template="")
        print('self.df_logs.iloc[23]:\n{}'.format(self.df_logs.iloc[23]))
        self.num_total_logs = len(self.df_logs)
        self.num_processed_logs = 0

    def represent(self):
        raise NotImplementedError

    def clustering(self):
        raise NotImplementedError

    def prepare_save_df(self):
        self.df_logs.assign(NewEventId="")
        self.original_df_logs = pd.read_csv(self.log_path)
        print(
            f"Original df_logs: {self.original_df_logs.shape}, Clustering df_logs: {self.df_logs.shape}"
        )
        templates_set = []
        for i, row in self.df_logs.iterrows():
            if row['Template'] in templates_set:
                template_id = templates_set.index(row['Template']) + 1
            else:
                templates_set.append(row['Template'])
                template_id = len(templates_set)
            self.df_logs.loc[i, 'NewEventId'] = f"E{template_id}"
        df = self.df_logs[["LineId", "Content", "NewEventId", "Template"]]
        # assign the row "Content" of original df_logs to df
        df["Content"] = self.original_df_logs["Content"]
        df.columns = ["LineId", "Content", "EventId", "EventTemplate"]
        return df

    def update_logs_with_map(self, template, child_id):
        if template == "":
            print("Fail to update Template is empty")
            return [], 0, {}
        parent_id = self.update_map_child2parent[child_id]
        bucket_ids_to_check = self.update_map_parent2child[parent_id]
        index = []
        all_indexes = {}
        total_matched = 0
        total_num_before, total_num_after = 0, 0
        for bucket_id in bucket_ids_to_check:
            current_logs_bucket = self.clusters[bucket_id]
            num_berfore = len(current_logs_bucket)
            current_logs_bucket.loc[:, "Matched"] = current_logs_bucket.apply(
                lambda row: verify_template_for_log_with_first_token(
                    row["Content"], template),
                axis=1)
            index = current_logs_bucket[current_logs_bucket["Matched"] ==
                                        True].index
            self.num_processed_logs += len(index)
            self.df_logs.loc[index, "Template"] = template
            current_logs_bucket = current_logs_bucket.loc[
                current_logs_bucket["Matched"] == False]
            self.clusters[bucket_id] = current_logs_bucket
            num_after = len(current_logs_bucket)
            total_matched += num_berfore - num_after
            total_num_before += num_berfore
            total_num_after += num_after
            all_indexes[bucket_id] = index.tolist()
        empty_bucket_num = len(
            [i for i in self.clusters.values() if len(i) != 0])
        print(
            f"[UpdateBucket] Logs: This iter found: {total_matched}, total: {self.num_processed_logs}/{self.num_total_logs}, "
            f"remain: {self.num_total_logs-self.num_processed_logs}. ")
        print(
            f"[UpdateBucket] Buckets: Checked {len(bucket_ids_to_check)} ({bucket_ids_to_check}), Parent Bucket size: {total_num_before} -> {total_num_after}, remain buckets: {empty_bucket_num}"
        )
        if total_matched == 0:
            return False, 0, {}
        return True, total_matched, all_indexes

    def update_logs_by_indexes(self, template, child_id, all_indexes):
        if template == "":
            print(
                "[TemplateBaseUpdate] Fail to modify Template from an empty template"
            )
            return 0
        if not all_indexes:
            print(
                "[TemplateBaseUpdate] No existing indexes to check and update")
            return 0
        parent_id = self.update_map_child2parent[child_id]
        bucket_ids_to_check = self.update_map_parent2child[parent_id]
        total, total_updated = 0, 0
        for bucket_id in bucket_ids_to_check:
            index = pd.Index(all_indexes[bucket_id])
            rows_to_process = self.df_logs.loc[index]
            verify_results = rows_to_process.apply(
                lambda row: verify_template_for_log_with_first_token(
                    row["Content"], template),
                axis=1)
            index_to_update = verify_results[verify_results == True].index
            self.df_logs.loc[index_to_update, "Template"] = template
            total_updated += len(index_to_update)
            total += len(index)
        print(
            f"[TemplateBaseUpdate] Update previous logs with merged template, succeed/all: {total_updated}/{total}, in child Bucket {bucket_ids_to_check}"
        )
        return total_updated

    def non_emtpy(self, hyperbucket_ID):
        children = self.update_map_child2parent[hyperbucket_ID]
        for k in children:
            if len(self.cluster[k]) > 0:
                return True
        return False

    def sample_hyperbucket(self, hyperbucket_ID):
        children = self.update_map_child2parent[hyperbucket_ID]
        current_logs_bucket_id = max(children,
                                     key=lambda i: len(self.cluster[i]))
        current_logs_bucket = self.clusters[current_logs_bucket_id]
        print(
            f"Sample from current logs bucket: ID: {current_logs_bucket_id}, Len: {current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )

        if len(current_logs_bucket) == 1:
            print('B')
            logs = current_logs_bucket["Content"].tolist()
            cluster_id = current_logs_bucket["cid2"].iloc[0]
            return cluster_id, logs
        else:
            print('C')
            assert len(current_logs_bucket) > 1
            candidate_logs = current_logs_bucket["Content"].drop_duplicates(
            ).tolist()
            print("len(candidate_logs): {}".format(len(candidate_logs)))
            cluster_id = current_logs_bucket["cid2"].iloc[0]
            return cluster_id, least_similar(candidate_logs, 5)


class TopKTokenClustering(BaseClustering):
    """ Very similar to Drain with a depth 5"""

    def __init__(self,
                 sample_method="lcu_sampling",
                 sample_size=3,
                 cluster_topk=3,
                 min_cluster_size=100,
                 sample_min_similarity=0.5,
                 lcu_lamb=0.5,
                 lcu_sample_size=3,
                 sample_size_auto="fixed",
                 add_regex="add",
                 regex=[],
                 add_skip_sim=False,
                 pad_query=True):
        super(TopKTokenClustering,
              self).__init__(sample_method,
                             sample_size,
                             min_cluster_size=min_cluster_size,
                             sample_min_similarity=sample_min_similarity,
                             lcu_lamb=lcu_lamb,
                             lcu_sample_size=lcu_sample_size,
                             sample_size_auto=sample_size_auto,
                             add_regex=add_regex,
                             regex=regex,
                             add_skip_sim=add_skip_sim,
                             pad_query=pad_query)
        self.cluster_topk = cluster_topk
        self.token_frequency = Counter()

    def represent(self):
        self.df_logs["length"] = self.df_logs["Content"].apply(
            get_tokens_length)
        self.log_lengths = self.df_logs["length"].tolist()

    def clustering(self):
        if len(self.log_lengths) == 0:
            self.represent()
        df_logs = self.df_logs[self.df_logs["Template"] == ""]
        grouped = df_logs.groupby("length").groups
        self.max_log_length, self.min_log_length = max(grouped.keys()), min(
            grouped.keys())

        # Cluster by log length
        _bucket_to_merge = {}
        for idx, key in enumerate(sorted(grouped.keys())):
            this_bucket = self.df_logs.iloc[grouped[key]]
            # debug
            #if key == 3:
            #    print(this_bucket)
            #    print(idx)
            #    exit(0)
            _bucket_to_merge[idx] = this_bucket
        self.clusters = _bucket_to_merge
        print(f"Clustering by log length: {len(self.clusters)}")
        print(
            f"Clustering by log length: {[len(i) for i in self.clusters.values()]}"
        )
        #print(self.clusters.keys())
        #print(type(self.clusters[0]))
        #exit(0)

        # Cluster by top-k tokens
        flat_clusters = {}
        for idx, cluster in self.clusters.items():
            #_clusters = self.clustering_by_topk_tokens(cluster)
            _clusters = self.brain_cluster(cluster)
            if idx == 2:
                # debug
                for _ in _clusters:
                    print('#' * 50)
                    print(_["Content"].drop_duplicates())
                #exit(0)
            cid2 = len(flat_clusters)
            for i, df in enumerate(_clusters):
                df.loc[:, "cid1"] = [idx] * len(df)
                df.loc[:, "cid2"] = [cid2 + i] * len(df)
            for child_idx in range(len(flat_clusters),
                                   len(flat_clusters) + len(_clusters)):
                self.update_map_child2parent[child_idx] = idx
            self.update_map_parent2child[idx] = list(
                range(len(flat_clusters),
                      len(flat_clusters) + len(_clusters)))
            for _clus in _clusters:
                flat_clusters[len(flat_clusters)] = _clus
            # print(f"- Clustering by content similarity (group-{idx}): {len(_clusters)}")
        print(
            f"Clustering (min_cluster_size={self.min_cluster_size}) by length and 1st 3 tokens: {len(flat_clusters)} clusters"
        )
        self.clusters = flat_clusters

        # Merge small clusters
        print(
            f"Clustering results: {[len(i) for i in self.clusters.values()]}")

        return self.clusters

    def brain_cluster(self, df):
        # 1. Split "Content" into tokens
        token_lists = df["Content"].str.split()

        # Assert same token length
        lengths = token_lists.str.len()
        assert lengths.nunique(
        ) == 1, "All rows must have same number of tokens"

        # 2. Build token matrix for ALL rows (original df)
        token_df = pd.DataFrame(token_lists.tolist())

        # 3. Build token matrix for UNIQUE Content only (for frequency computation)
        unique_content = df["Content"].drop_duplicates()
        # debug
        t = lengths.iloc[0]
        if False and t == 3:
            print('here')
            print(len(unique_content))
            print(unique_content)
        unique_token_lists = unique_content.str.split()
        unique_token_df = pd.DataFrame(unique_token_lists.tolist())

        # 4. Compute column-wise token frequencies ONLY on unique rows
        freq_lookup = {}
        for col in unique_token_df.columns:
            freq_lookup[col] = unique_token_df[col].value_counts().to_dict()
            #freq_lookup[col] = token_df[col].value_counts().to_dict()

        # 5. Map frequencies back to ALL rows (including duplicates)
        freq_df = pd.DataFrame({
            col: token_df[col].map(freq_lookup[col])
            for col in token_df.columns
        })
        if False and t == 3:
            x_df = pd.DataFrame({
                col: unique_token_df[col].map(freq_lookup[col])
                for col in token_df.columns
            })
            freq_df = x_df
            print(x_df)

        # 6. For each row (in original df), compute the feature tokens
        features = []
        for i, row in freq_df.iterrows():
            freqs = row.tolist()
            # most common frequency value
            #most_common_freq = Counter(freqs).most_common(1)[0][0]
            if True:
                freq_counts = Counter(freqs)
                max_count = max(freq_counts.values())
                tied_freqs = [
                    f for f, c in freq_counts.items() if c == max_count
                ]
                most_common_freq = max(tied_freqs)
                if most_common_freq == 1:
                    most_common_freq = max(freqs)
            if False and t == 3:
                print(freqs)
                print(most_common_freq)
            # select tokens whose freq == most_common_freq
            tokens = [
                token_df.iloc[i, j] for j, f in enumerate(freqs)
                if f == most_common_freq
            ]
            features.append(
                tuple(tokens))  # tuple makes it hashable for grouping

        # 7. Attach feature to df
        #if t == 3: exit(0)
        df = df.copy()
        df["_feature"] = features

        # 8. Split df by identical features
        grouped_dfs = [
            #group.drop(columns="_feature")
            group for _, group in df.groupby("_feature")
        ]

        # lezhang.thu - start
        #groups = [group for _, group in df.groupby("_feature")]

        #singleton_groups = [g for g in groups if len(g) == 1]
        #multi_groups = [g for g in groups if len(g) > 1]

        #if singleton_groups:
        #    merged_singletons = pd.concat(singleton_groups, ignore_index=True)
        #    multi_groups.append(merged_singletons)

        #grouped_dfs = [g.drop(columns="_feature") for g in multi_groups]
        # lezhang.thu - end

        return grouped_dfs


def get_tokens(log, separator=[" "]):
    for sep in separator:
        log = log.replace(sep, " ")
    return log.split()


def get_tokens_length(log, separator=[" "]):
    return len(get_tokens(log, separator))


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def least_similar(candidate_logs, n_anchors=5):
    if len(candidate_logs) <= 1:
        return candidate_logs

    n = len(candidate_logs)
    anchors = [candidate_logs[0]]
    selected_indices = {0}

    # initialize min_sims with similarity to the first anchor
    min_sims = calculate_jaccard_one_to_many(candidate_logs[0], candidate_logs)
    min_sims[0] = math.inf  # prevent re-selecting anchor 0

    def random_argmin(values):
        min_val = min(values)
        candidates = [i for i, v in enumerate(values) if v == min_val]
        return random.choice(candidates)

    for _ in range(1, min(n_anchors, n)):
        # find least similar log (lowest min similarity)
        #next_idx = min(range(n), key=lambda i: min_sims[i])
        next_idx = random_argmin(min_sims)
        anchors.append(candidate_logs[next_idx])
        selected_indices.add(next_idx)

        # compute similarity to the new anchor
        sims_new = calculate_jaccard_one_to_many(candidate_logs[next_idx],
                                                 candidate_logs)

        # update min similarities
        for i in range(n):
            if i in selected_indices:
                min_sims[i] = math.inf
            else:
                min_sims[i] = min(min_sims[i], sims_new[i])

    return anchors
