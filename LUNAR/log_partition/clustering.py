import re
import math
import random
import itertools
import pandas as pd
from collections import Counter

pd.set_option('mode.chained_assignment', None)
from LUNAR.utils import verify_template_for_log_regex, verify_template_for_log_with_first_token, verify_template_for_log_with_first_token_subset
from LUNAR.utils import preprocess_log_for_query
from LUNAR.log_partition.text_distance import similarity_jaccard_words, calculate_same_one_to_many, calculate_jaccard_one_to_many, calculate_jaccard_one_to_many_mask
from LUNAR.log_partition.text_distance import calculate_jaccard_and_diff_self_all_comp

#CHECK_CLUSTERS =[161, ]


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
        self.vectors = []
        self.log_lengths = []
        # self.clusters = []
        self.clusters = {}
        self.current_logs_bucket_id = -1
        self.current_logs_bucket = pd.DataFrame(
            columns=["LineId", "Content", "EventId", "Template"])
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

    def get_lookup_table(self):
        lookup_table = {
            row["Content"]: row['EventTemplate']
            for rid, row in self.df_logs.iterrows()
        }
        return lookup_table

    def represent(self):
        raise NotImplementedError

    def clustering(self):
        raise NotImplementedError

    def sample_for_llm(self):
        raise NotImplementedError

    def get_cluster(self, method=""):
        return self.clusters[self.current_logs_bucket_id]

    def get_current_bucket(self):
        return self.current_logs_bucket

    def get_bucket_num(self):
        return len(self.clusters)

    def get_current_bucket_depth(self):
        return len(self.current_logs_bucket)

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

    def update_logs(self, template):
        if template == "":
            print("Fail to update Template is empty")
            return [], 0
        num_berfore = len(self.current_logs_bucket)
        self.current_logs_bucket.loc[:,
                                     "Matched"] = self.current_logs_bucket.apply(
                                         lambda row:
                                         verify_template_for_log_with_first_token(
                                             row["Content"], template),
                                         axis=1)

        index = self.current_logs_bucket[self.current_logs_bucket["Matched"] ==
                                         True].index

        self.num_processed_logs += len(index)
        self.df_logs.loc[index, "Template"] = template
        self.current_logs_bucket = self.current_logs_bucket.loc[
            self.current_logs_bucket["Matched"] == False]
        self.clusters[self.current_logs_bucket_id] = self.current_logs_bucket
        num_after = len(self.current_logs_bucket)
        empty_bucket_num = len(
            [i for i in self.clusters.values() if len(i) != 0])
        print(
            f"[UpdateBucket] This iter found: {len(index)}, total: {self.num_processed_logs}/{self.num_total_logs}, "
            f"remain: {self.num_total_logs-self.num_processed_logs}. Bucket size: {num_berfore} -> {num_after}, remain buckets: {empty_bucket_num}"
        )
        if num_berfore == num_after:
            return False, 0
        return True, len(index)

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

    def sample_by_cluster_size(self, dedup=True):
        if len(self.clusters) == 0:
            self.clustering()
        # Strategy: always sample the largest cluster
        max_cluster_id = max(self.clusters,
                             key=lambda k: len(self.clusters[k]))
        self.current_logs_bucket_id = max_cluster_id
        self.current_logs_bucket = self.clusters[self.current_logs_bucket_id]
        print(
            f"Sample {self.sample_size} from current logs bucket: ID: {self.current_logs_bucket_id}, Len: {self.current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(self.current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )

        if len(self.current_logs_bucket) == 0:
            self.clusters.pop(self.current_logs_bucket_id)
            cluster_id, sampled = self.sample_by_cluster_size(dedup=dedup)
            return cluster_id, sampled
        else:
            logs = self.current_logs_bucket["Content"].tolist()
            if self.sample_size_auto:
                length_this_bucket = self.current_logs_bucket["length"].iloc[0]
                self.sample_size = compute_adaptive_sample_size(
                    length_this_bucket, logs[0], self.sample_size_assigned)

            if dedup:
                logs = remove_duplicates(logs)
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, sampling_from_list(logs,
                                                  self.sample_size,
                                                  padding=self.pad_query)

    def sample_by_max_same(self, dedup=True):
        if len(self.clusters) == 0:
            self.clustering()
        # Strategy: always sample the largest cluster
        max_cluster_id = max(self.clusters,
                             key=lambda k: len(self.clusters[k]))
        self.current_logs_bucket_id = max_cluster_id
        self.current_logs_bucket = self.clusters[self.current_logs_bucket_id]
        print(
            f"Sample {self.sample_size} from current logs bucket: ID: {self.current_logs_bucket_id}, Len: {self.current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(self.current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )

        if len(self.current_logs_bucket) == 0:
            self.clusters.pop(self.current_logs_bucket_id)
            cluster_id, sampled = self.sample_by_max_same(dedup=dedup)
            return cluster_id, sampled
        elif len(self.current_logs_bucket) == 1:
            logs = self.current_logs_bucket["Content"].tolist()
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, sampling_from_list(logs,
                                                  1,
                                                  padding=self.pad_query)
        else:
            anchor_log = self.current_logs_bucket["Content"].iloc[0]
            if self.sample_size_auto:
                length_this_bucket = self.current_logs_bucket["length"].iloc[0]
                self.sample_size = compute_adaptive_sample_size(
                    length_this_bucket, anchor_log, self.sample_size_assigned)

            candidate_logs = self.current_logs_bucket["Content"].iloc[
                1:].tolist()
            similarities = calculate_same_one_to_many(anchor_log,
                                                      candidate_logs)
            sorted_logs = sorted(zip(similarities, candidate_logs),
                                 key=lambda x: x[0],
                                 reverse=True)
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, [anchor_log] + [
                log for sim, log in sorted_logs[:self.sample_size - 1]
            ]

    def sample_by_minmax_jaccard(self, dedup=True):
        if len(self.clusters) == 0:
            self.clustering()
        # Strategy: always sample the largest cluster
        max_cluster_id = max(self.clusters,
                             key=lambda k: len(self.clusters[k]))
        self.current_logs_bucket_id = max_cluster_id
        self.current_logs_bucket = self.clusters[self.current_logs_bucket_id]
        print(
            f"Sample {self.sample_size} from current logs bucket: ID: {self.current_logs_bucket_id}, Len: {self.current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(self.current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )

        if len(self.current_logs_bucket) == 0:
            self.clusters.pop(self.current_logs_bucket_id)
            cluster_id, sampled = self.sample_by_minmax_jaccard(dedup=dedup)
            return cluster_id, sampled
        elif len(self.current_logs_bucket) == 1:
            logs = self.current_logs_bucket["Content"].tolist()
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, sampling_from_list(logs,
                                                  1,
                                                  padding=self.pad_query)
        else:
            anchor_log, candidate_logs = self.anchor_log_selection(
                self.current_logs_bucket["Content"].tolist(), method="first")
            if self.sample_size_auto:
                length_this_bucket = self.current_logs_bucket["length"].iloc[0]
                self.sample_size = compute_adaptive_sample_size(
                    length_this_bucket, anchor_log, self.sample_size_assigned)

            if dedup:
                candidate_logs = remove_duplicates(candidate_logs)
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            sampled = sampling_from_sorted_list(
                anchor_log,
                candidate_logs,
                self.sample_size - 1,
                add_skip_sim=self.add_skip_sim,
                min_sim_threshold=self.sample_min_similarity,
                remove_same=True,
            )

            return cluster_id, sampled

    def sample_by_lcu_sampling(self, dedup=True):
        if len(self.clusters) == 0:
            self.clustering()
        # Strategy: always sample the largest cluster
        max_cluster_id = max(self.clusters,
                             key=lambda k: len(self.clusters[k]))
        # debug
        if True:
            max_cluster_id = 238 
            print(self.clusters[max_cluster_id]['_feature'].iloc[0])
            print(self.clusters[max_cluster_id].drop_duplicates(
                subset='Content', keep='last'))
            print("self.pad_query: {}".format(self.pad_query))
            #exit(0)

        self.current_logs_bucket_id = max_cluster_id
        self.current_logs_bucket = self.clusters[self.current_logs_bucket_id]
        proposal_template = self.current_logs_bucket["_feature"].iloc[0]
        print(
            f"Sample {self.sample_size} from current logs bucket: ID: {self.current_logs_bucket_id}, Len: {self.current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(self.current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )
        # debug
        #self.clusters[4].to_csv('cluster-4.csv', index=False)
        #exit(0)

        if len(self.current_logs_bucket) == 0:
            print('A')
            assert False, "The branch should NOT be entered!"
            self.clusters.pop(self.current_logs_bucket_id)
            cluster_id, sampled = self.sample_by_lcu_sampling(dedup=dedup)
            return cluster_id, sampled, proposal_template
        elif len(self.current_logs_bucket) == 1:
            print('B')
            logs = self.current_logs_bucket["Content"].tolist()
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, logs, None
        else:
            print('C')
            candidate_logs = self.current_logs_bucket[
                "Content"].drop_duplicates().tolist()
            print("len(candidate_logs): {}".format(len(candidate_logs)))
            if len(candidate_logs) == 1:
                proposal_template = None
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            #return cluster_id, candidate_logs[:5], proposal_template
            #return cluster_id, candidate_logs[:5], None
            return cluster_id, least_similar(candidate_logs), None

            #anchor_log, candidate_logs = self.anchor_log_selection(
            #    self.current_logs_bucket["Content"].tolist(), method="first")
            #print("len(candidate_logs): {}".format(len(candidate_logs)))
            ##print("self.sample_size_auto: {}".format(self.sample_size_auto))
            #print("self.sample_size: {}".format(self.sample_size))
            ##if False:
            #if True:
            #    if self.sample_size_auto:
            #        length_this_bucket = self.current_logs_bucket[
            #            "length"].iloc[0]
            #        #self.sample_size = compute_adaptive_sample_size(length_this_bucket, anchor_log, self.sample_size_assigned)
            #        self.sample_size = compute_adaptive_sample_size(
            #            length_this_bucket, anchor_log, 5)
            #        print("self.sample_size: {}".format(self.sample_size))
            #    # debug
            #    #self.sample_size = max(3, self.sample_size)
            #if dedup:
            #    candidate_logs = remove_duplicates(candidate_logs)
            #    print("len(candidate_logs): {}".format(len(candidate_logs)))
            #cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            #if False:
            #    sampled = sampling_from_sorted_list(
            #        anchor_log,
            #        candidate_logs,
            #        self.sample_size - 1,
            #        lcu_lamb=self.lcu_lamb,
            #        lcu_sample_size=self.lcu_sample_size,
            #        add_skip_sim=self.add_skip_sim,
            #        min_sim_threshold=self.sample_min_similarity,
            #        remove_same=True,
            #    )
            #sampled = candidate_logs[:self.sample_size]
            ##sampled = candidate_logs[:1]
            #return cluster_id, sampled

    def sample_by_lcu_sampling_parallel(self, input_clusters, dedup=True):
        if not input_clusters:
            print("No clusters found")
            return
        # Strategy: always sample the largest cluster
        max_cluster_id = max(input_clusters,
                             key=lambda k: len(input_clusters[k]))
        current_logs_bucket_id = max_cluster_id
        current_logs_bucket = input_clusters[current_logs_bucket_id]
        print(
            f"Sample {self.sample_size} from current logs bucket: ID: {self.current_logs_bucket_id}, Len: {self.current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(self.current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )

        if len(current_logs_bucket) == 1:
            logs = current_logs_bucket["Content"].tolist()
            cluster_id = current_logs_bucket["cid2"].iloc[0]
            return cluster_id, sampling_from_list(logs,
                                                  self.sample_size,
                                                  padding=self.pad_query)
        else:
            anchor_log, candidate_logs = self.anchor_log_selection(
                current_logs_bucket["Content"].tolist(), method="first")
            if self.sample_size_auto:
                length_this_bucket = self.current_logs_bucket["length"].iloc[0]
                self.sample_size = compute_adaptive_sample_size(
                    length_this_bucket, anchor_log, self.sample_size_assigned)

            if dedup:
                candidate_logs = remove_duplicates(candidate_logs)
            cluster_id = current_logs_bucket["cid2"].iloc[0]
            sampled = sampling_from_sorted_list(
                anchor_log,
                candidate_logs,
                self.sample_size - 1,
                lcu_lamb=self.lcu_lamb,
                lcu_sample_size=self.lcu_sample_size,
                add_skip_sim=self.add_skip_sim,
                min_sim_threshold=self.sample_min_similarity,
                remove_same=True,
            )

            return cluster_id, sampled

    def anchor_log_selection(self, logs, method="first"):
        if method == "first":
            return logs[0], logs[1:]
        elif method == "random":
            idx = random.randint(1, len(logs) - 1)
            return logs[idx], logs[:idx] + logs[idx + 1:]
        else:
            return logs[0], logs[1:]


#class TopKTokenClusteringParallel(BaseClustering):
#    """ Very similar to Drain with a depth 5"""
#
#    def __init__(self, sample_method="lcu_sampling", sample_size=3, cluster_topk=3, min_cluster_size=100, sample_min_similarity=0.5,
#                 lcu_lamb=0.5, lcu_sample_size=3, sample_size_auto="fixed", add_regex="add", regex=[],
#                 add_skip_sim=False, pad_query=True):
#        super(TopKTokenClusteringParallel, self).__init__(sample_method, sample_size,
#                                                  min_cluster_size=min_cluster_size,
#                                                  sample_min_similarity=sample_min_similarity,
#                                                  lcu_lamb=lcu_lamb,
#                                                  lcu_sample_size=lcu_sample_size,
#                                                  sample_size_auto=sample_size_auto,
#                                                  add_regex=add_regex,
#                                                  regex=regex,
#                                                  add_skip_sim=add_skip_sim,
#                                                  pad_query=pad_query)
#        self.cluster_topk = cluster_topk
#        self.token_frequency = Counter()
#        self.vocab = Vocab()
#
#    def represent(self):
#        self.df_logs["length"] = self.df_logs["Content"].apply(get_tokens_length)
#        self.log_lengths = self.df_logs["length"].tolist()
#        # self.vocab.build(self.df_logs["Content"].tolist())
#
#    def clustering(self):
#        if len(self.log_lengths) == 0:
#            self.represent()
#        df_logs = self.df_logs[self.df_logs["Template"] == ""]
#        grouped = df_logs.groupby("length").groups
#        self.max_log_length, self.min_log_length = max(grouped.keys()), min(grouped.keys())
#
#        # Cluster by log length
#        _bucket_to_merge = {}
#        for idx, key in enumerate(sorted(grouped.keys())):
#            this_bucket = self.df_logs.iloc[grouped[key]]
#            _bucket_to_merge[idx] = this_bucket
#        self.clusters = _bucket_to_merge
#        print(f"Clustering by log length: {len(self.clusters)}")
#        print(f"Clustering by log length: {[len(i) for i in self.clusters.values()]}")
#
#        # Cluster by top-k tokens
#        flat_clusters = {}
#        for idx, cluster in self.clusters.items():
#            _clusters = self.clustering_by_topk_tokens(cluster)
#            cid2 = len(flat_clusters)
#            for i, df in enumerate(_clusters):
#                df.loc[:, "cid1"] = [idx] * len(df)
#                df.loc[:, "cid2"] = [cid2+i] * len(df)
#            for child_idx in range(len(flat_clusters), len(flat_clusters)+len(_clusters)):
#                self.update_map_child2parent[child_idx] = idx
#            self.update_map_parent2child[idx] = list(range(len(flat_clusters), len(flat_clusters)+len(_clusters)))
#            for _clus in _clusters:
#                flat_clusters[len(flat_clusters)] = _clus
#            # print(f"- Clustering by content similarity (group-{idx}): {len(_clusters)}")
#            a = 1
#        print(f"Clustering (min_cluster_size={self.min_cluster_size}) by length and 1st 3 tokens: {len(flat_clusters)} clusters")
#        self.clusters = flat_clusters
#
#        # Merge small clusters
#        print(f"Clustering results: {[len(i) for i in self.clusters.values()]}")
#
#        return self.clusters
#
#    def sample_for_llm(self, buckets=None):
#        if not buckets:
#            return self.sample_by_lcu_sampling(dedup=True)
#        else:
#            return self.sample_by_lcu_sampling_parallel(buckets, dedup=True)
#
#    def clustering_by_topk_tokens(self, log_df):
#        clusters = []
#        vocab = Vocab()
#        vocab.build(log_df["Content"].tolist())
#
#        topk_map_rows = {}
#        for rid, row in log_df.iterrows():
#            topk = vocab.topk_tokens(row["Content"], self.cluster_topk)
#            if topk in topk_map_rows:
#                topk_map_rows[topk].append(row)
#            else:
#                topk_map_rows[topk] = [row]
#
#        n_topk = self.cluster_topk
#        while len(topk_map_rows) > 0:
#            topk_tokens = list(topk_map_rows.keys())
#            map_long_to_short = {toks: toks[:n_topk] for i, toks in enumerate(topk_tokens)}
#
#            cluster_map = {}
#            for toks_long in topk_tokens:
#                tok_short = map_long_to_short[toks_long]
#                if tok_short in cluster_map:
#                    cluster_map[tok_short].append(toks_long)
#                else:
#                    cluster_map[tok_short] = [toks_long]
#
#            a=0
#            for toks_short, toks_longs in cluster_map.items():
#                grouping_rows = [topk_map_rows[toks_long] for toks_long in toks_longs]
#                grouping_rows = [j for i in grouping_rows for j in i]
#                if len(grouping_rows) < self.min_cluster_size:
#                    continue
#                else:
#                    clusters.append(pd.DataFrame(grouping_rows))
#                    for toks_long in toks_longs:
#                        del topk_map_rows[toks_long]
#
#            n_topk = n_topk - 1
#            if n_topk == 0 and len(topk_map_rows) > 0:
#                merged_row = pd.DataFrame([line for lines in topk_map_rows.values() for line in lines])
#                clusters.append(merged_row)
#                break
#        size_min = min([len(i) for i in clusters]) if len(clusters) > 0 else 0
#        size_max = max([len(i) for i in clusters]) if len(clusters) > 0 else 0
#        print(f"--- Finished 2nd Level Clustering by top-{self.cluster_topk} tokens: {len(clusters)} clusters, "
#              f"Total/Min/Max: {len(log_df)}/{size_min}/{size_max} Logs. Last cluster logs: {len(clusters[-1])}")
#        return clusters


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
        self.vocab = Vocab()

    def represent(self):
        self.df_logs["length"] = self.df_logs["Content"].apply(
            get_tokens_length)
        self.log_lengths = self.df_logs["length"].tolist()
        # self.vocab.build(self.df_logs["Content"].tolist())

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

    def sample_for_llm(self):
        return self.sample_by_lcu_sampling(dedup=True)

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

        return grouped_dfs

    #def brain_cluster(self, df):
    #    # 1. Split "Content" into tokens
    #    token_lists = df["Content"].str.split()
    #    #print('debug')
    #    #for _ in token_lists[:10]:
    #    #    print(_)
    #    #print("len(token_lists): {}".format(len(token_lists)))
    #    # Assert same token length
    #    lengths = token_lists.str.len()
    #    assert lengths.nunique() == 1, "All rows must have same number of tokens"
    #    n_cols = lengths.iloc[0]
    #    # 2. Build token matrix
    #    token_df = pd.DataFrame(token_lists.tolist())
    #    # 3. Compute column-wise token frequencies
    #    freq_df = pd.DataFrame({
    #        col: token_df[col].map(token_df[col].value_counts()) for col in token_df.columns
    #    })
    #    #print(token_df)
    #    #print(freq_df)
    #    # 4. For each row, compute the feature tokens
    #    features = []
    #    for i, row in freq_df.iterrows():
    #        freqs = row.tolist()
    #        # most common frequency value
    #        most_common_freq = Counter(freqs).most_common(1)[0][0]
    #        # select tokens whose freq == most_common_freq
    #        tokens = [
    #            token_df.iloc[i, j] for j, f in enumerate(freqs) if f == most_common_freq
    #        ]
    #        features.append(tuple(tokens))  # tuple makes it hashable for grouping
    #    # Attach feature to df
    #    df = df.copy()
    #    df["_feature"] = features
    #    #print(df)
    #    # 5. Split df by identical features
    #    grouped_dfs = [group.drop(columns="_feature") for _, group in df.groupby("_feature")]
    #    #for idx, _ in enumerate(grouped_dfs):
    #    #    print('group_{}'.format(idx))
    #    #    print(_)
    #    #exit(0)
    #    return grouped_dfs

    def clustering_by_topk_tokens(self, log_df, verbose=False):
        clusters = []
        vocab = Vocab()
        vocab.build(log_df["Content"].tolist())

        topk_map_rows = {}
        for rid, row in log_df.iterrows():
            topk = vocab.topk_tokens(row["Content"], self.cluster_topk)
            if topk in topk_map_rows:
                topk_map_rows[topk].append(row)
            else:
                topk_map_rows[topk] = [row]

        n_topk = self.cluster_topk
        while len(topk_map_rows) > 0:
            topk_tokens = list(topk_map_rows.keys())
            map_long_to_short = {
                toks: toks[:n_topk]
                for i, toks in enumerate(topk_tokens)
            }

            cluster_map = {}
            for toks_long in topk_tokens:
                tok_short = map_long_to_short[toks_long]
                if tok_short in cluster_map:
                    cluster_map[tok_short].append(toks_long)
                else:
                    cluster_map[tok_short] = [toks_long]

            a = 0
            for toks_short, toks_longs in cluster_map.items():
                grouping_rows = [
                    topk_map_rows[toks_long] for toks_long in toks_longs
                ]
                grouping_rows = [j for i in grouping_rows for j in i]
                if len(grouping_rows) < self.min_cluster_size:
                    continue
                else:
                    clusters.append(pd.DataFrame(grouping_rows))
                    for toks_long in toks_longs:
                        del topk_map_rows[toks_long]

            n_topk = n_topk - 1
            if n_topk == 0 and len(topk_map_rows) > 0:
                merged_row = pd.DataFrame([
                    line for lines in topk_map_rows.values() for line in lines
                ])
                clusters.append(merged_row)
                break
        size_min = min([len(i) for i in clusters]) if len(clusters) > 0 else 0
        size_max = max([len(i) for i in clusters]) if len(clusters) > 0 else 0
        print(
            f"--- Finished 2nd Level Clustering by top-{self.cluster_topk} tokens: {len(clusters)} clusters, "
            f"Total/Min/Max: {len(log_df)}/{size_min}/{size_max} Logs. Last cluster logs: {len(clusters[-1])}"
        )
        return clusters


import calendar
from collections import OrderedDict


class Vocab:

    def __init__(self, stopwords=["<*>"]):
        stopwords = [
            "a",
            "an",
            "and",
            "i",
            "ie",
            "so",
            "to",
            "the",

        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))

    def build(self, sequences):
        # print("Build vocab with examples: ", len(sequences))
        sequences = [get_tokens(sequence) for sequence in sequences]
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            #print(sequence)
            self.update(sequence)

    def update(self, sequence):
        sequence = self.__filter_stopwords(sequence)
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = get_tokens(sequence)
        sequence = self.__filter_stopwords(sequence)

        token_count = [(token, self.token_counter[token])
                       for token in sequence]
        unique_tokens = list(OrderedDict.fromkeys(token_count))
        sorted_tokens = sorted(unique_tokens, key=lambda x: x[1], reverse=True)
        topk_keys = tuple(sorted_tokens[:topk])

        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]


def get_tokens(log, separator=[" "]):
    for sep in separator:
        log = log.replace(sep, " ")
    return log.split()


def get_tokens_length(log, separator=[" "]):
    return len(get_tokens(log, separator))


def judge_var_log(log, placeholder="<*>"):
    log = log.replace(placeholder, "")
    alphabet_set = set(
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!")
    if set(log).issubset(alphabet_set):
        return False
    else:
        return True


def compute_adaptive_sample_size(length_log,
                                 anchor_log,
                                 max_size=3,
                                 src_size=[1, 40]):
    tgt_size = [1, 5]
    with_potential_var = judge_var_log(anchor_log)
    if with_potential_var:
        return max_size
    #elif length_log <= 5:
    elif length_log <= 2:
        return 1
    else:
        return max_size


def replace_common_params(log):
    re_url = r"^(https?:\/\/)?([a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5})(:[0-9]{1,5})?(\/.*)?$"
    replaced_log = re.sub(re_url, "<*>", log)
    re_ip = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    replaced_log = re.sub(re_ip, "<*>", replaced_log)
    re_time = r"^(1[0-2]|0?[1-9]):[0-5][0-9] (AM|PM)$"
    replaced_log = re.sub(re_time, "<*>", replaced_log)
    re_phone = r"^\+?[1-9]\d{1,14}$"
    replaced_log = re.sub(re_phone, "<*>", replaced_log)
    re_date = r"^(?:19|20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])$"
    replaced_log = re.sub(re_date, "<*>", replaced_log)

    return replaced_log


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def anchor_log_selection(logs, method="first"):
    if method == "first":
        return logs[0], logs[1:]
    elif method == "random":
        idx = random.randint(1, len(logs) - 1)
        return logs[idx], logs[:idx] + logs[idx + 1:]
    else:
        return logs[0], logs[1:]


def pad_list(lst, size):
    if len(lst) < size:
        lst += [lst[i % len(lst)] for i in range(size - len(lst))]
    return lst


def least_similar(candidate_logs, n_anchors=5):
    if len(candidate_logs) <= 1:
        return candidate_logs

    n = len(candidate_logs)
    anchors = [candidate_logs[0]]
    selected_indices = {0}

    # initialize min_sims with similarity to the first anchor
    min_sims = calculate_jaccard_one_to_many(candidate_logs[0], candidate_logs)
    min_sims[0] = math.inf  # prevent re-selecting anchor 0

    for _ in range(1, min(n_anchors, n)):
        # find least similar log (lowest min similarity)
        next_idx = min(range(n), key=lambda i: min_sims[i])
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


#def least_similar(candidate_logs):
#    if len(candidate_logs) == 1:
#        return candidate_logs
#    anchor_log = candidate_logs[0]
#    candidate_logs = candidate_logs[1:]
#    # compute similarities
#    similarities = calculate_jaccard_one_to_many(anchor_log, candidate_logs)
#    zipped = list(zip(similarities, candidate_logs))
#    sorted_keys = sorted(zipped, key=lambda k: k[0])
#    x_extracted = [k[1] for k in sorted_keys[:4]]
#    return [anchor_log] + x_extracted


def sampling_from_sorted_list(anchor_log,
                              candidate_logs,
                              sample_size,
                              add_skip_sim=False,
                              min_sim_threshold=0.5,
                              lcu_lamb=0.2,
                              lcu_sample_size=8,
                              remove_same=True):
    if len(candidate_logs) == 0:
        return [anchor_log]
    if sample_size <= 0:
        return [anchor_log]

    print("within sampling_from_sorted_list, len(candidate_logs): {}".format(
        len(candidate_logs)))
    # compute similarities
    similarities = calculate_jaccard_one_to_many(anchor_log, candidate_logs)
    zipped = list(zip(similarities, candidate_logs))
    similarity_counter = Counter(similarities)

    if False:

        # remove totally different logs, with sim=0
        if 0.0 in similarity_counter.keys():
            zipped = [(sim, log) for sim, log in zipped if sim != 0.0]
            del similarity_counter[0.0]
        # remove the same logs, with sim=1
        if remove_same and 1.0 in similarity_counter.keys():
            zipped = [(sim, log) for sim, log in zipped if sim != 1.0]
            del similarity_counter[1.0]
        #if add_skip_sim:
        if False:
            skip_sim_threshold = 0.33
            zipped = [(sim, log) for sim, log in zipped
                      if sim >= skip_sim_threshold]
            to_remove = [
                val for val in similarity_counter.keys()
                if val < skip_sim_threshold
            ]
            for val in to_remove:
                del similarity_counter[val]

        if len(zipped) == 0:
            print(f"Sampling from {len(candidate_logs)} logs failed")
            return [anchor_log]

        # get a sim2logs
        sim2logs = {}
        for sim, log in zipped:
            if sim not in sim2logs:
                sim2logs[sim] = []
            sim2logs[sim].append(log)

        # remove logs below min similarity threshold.
        max_sim = max(similarity_counter.keys())
        max_sim_log = sim2logs[max_sim][0] if len(
            sim2logs[max_sim]) > 0 else ""
        print(
            f"Sampling from {len(zipped)} logs, Sim Level: {len(sim2logs)}, MaxSim to anchor: {max_sim:.4f}. Anchor: `{anchor_log}`, MaxSim Log: `{max_sim_log}`."
        )
    if False:
        #if True:
        if max_sim >= min_sim_threshold:
            zipped = [(sim, log) for sim, log in zipped
                      if sim >= min_sim_threshold]
            for val in similarity_counter.keys():
                if val < min_sim_threshold:
                    del sim2logs[val]
            if len(sim2logs) == 0:
                return [anchor_log]
        else:
            return [anchor_log]
    else:
        # debug - start
        sorted_keys = sorted(zipped, key=lambda k: k[0], reverse=True)
        x_extracted = [k[1] for k in sorted_keys[:lcu_sample_size]]
        print(x_extracted)
        return [anchor_log] + x_extracted
        # debug - end

    # LCU sampling:
    log_pools = [
        random.sample(_logs, min(lcu_sample_size, len(_logs)))
        for _sim, _logs in sim2logs.items()
    ]
    log_pools = [_log for _logs in log_pools for _log in _logs]
    if len(log_pools) < sample_size:
        return [anchor_log] + log_pools

    all_combinations = [
        list(i) for i in itertools.combinations(log_pools, sample_size)
    ]
    all_combinations = [[anchor_log] + i for i in all_combinations]
    lcu_scores = [
        calculate_jaccard_and_diff_self_all_comp(comb, lamb=lcu_lamb)
        for comb in all_combinations
    ]
    min_lcu_score = min(lcu_scores)
    all_min_lcu_score_combinations = [
        comb for comb, score in zip(all_combinations, lcu_scores)
        if score == min_lcu_score
    ]
    best_combination = all_min_lcu_score_combinations[0]
    # best_combination = all_combinations[np.argmax(lcu_scores)]
    return best_combination
