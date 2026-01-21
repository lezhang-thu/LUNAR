import numpy as np
import regex
from LUNAR.llm_module.post_process import post_process_template
from LUNAR.utils import validate_template, verify_template_for_log_with_first_token


def split_template_naive(template):
    """
    Split a log template into parts using the default space character.

    :param template: The log template to be split.
    :return: A list of parts obtained by splitting the template.
    """
    return template.split(" ")


def jaccard_similarity(parts1, parts2):
    """
    Calculate the Jaccard similarity between two sets of template parts.

    :param parts1: The first set of template parts.
    :param parts2: The second set of template parts.
    :return: The Jaccard similarity score.
    """
    common = set(parts1).intersection(parts2)
    union = set(parts1).union(parts2)
    return len(common) / len(union)


def merge_sorted_lists(list1, list2):
    """
    Merge two sorted lists.

    :param list1: The first sorted list.
    :param list2: The second sorted list.
    :return: The merged sorted list.
    """
    merged_list = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            if not merged_list or merged_list[-1] != list1[i]:
                merged_list.append(list1[i])
            i += 1
        elif list1[i] > list2[j]:
            if not merged_list or merged_list[-1] != list2[j]:
                merged_list.append(list2[j])
            j += 1
        else:
            if not merged_list or merged_list[-1] != list1[i]:
                merged_list.append(list1[i])
            i += 1
            j += 1

    while i < len(list1):
        if not merged_list or merged_list[-1] != list1[i]:
            merged_list.append(list1[i])
        i += 1

    while j < len(list2):
        if not merged_list or merged_list[-1] != list2[j]:
            merged_list.append(list2[j])
        j += 1

    return merged_list


class TemplateDatabase:
    """
    A class for managing a database of log templates.

    Attributes:
        None (as the __init__ method is currently empty).
    """

    def __init__(self):
        self.template_items = {}
        self.template_list = []

    def add_template(self, event_template, indexes={}, relevant_templates=[]):
        """
        Add a new template to the database.

        :param event_template: The log template to be added.
        :param indexes: A dictionary of indexes related to the template. Defaults to {}.
        :param relevant_templates: A list of relevant templates. Defaults to [].
        """
        template_tokens = split_template_naive(event_template)
        if not template_tokens or event_template == "<*>":
            return False, event_template, None, None
        if len(self.template_items) == 0 or len(template_tokens) == 1:
            self._insert_template(event_template, indexes)
            return False, event_template, None, None

        x_t = [split_template_naive(t) for t in self.template_list]
        coarse_similarities = [
            jaccard_similarity(template_tokens, t) for t in x_t
        ]

        # only compare with the most similar template
        max_sim_idx = np.argmax(coarse_similarities)
        xyz = self.template_list[max_sim_idx]

        new_template = self._judge_template_merge_combine(event_template, xyz)
        if new_template:
            print(f"[TemplateDB] Merge: `{event_template}` | `{xyz}`")
            insert_indexes = self._update_template(new_template, indexes,
                                                   max_sim_idx)
            print(f"[TemplateDB] Merged: -> `{new_template}`")
            return True, new_template, insert_indexes, xyz
        else:
            self._insert_template(event_template, indexes)
            return False, event_template, None, xyz

    def _judge_template_merge_combine(self, template1, template2):
        if template1 == template2:
            return template1
        import re

        def is_match(template, log):
            regex = re.escape(template)
            regex = regex.replace(r'<\*>', '.*?')
            regex = '^' + regex + '$'
            return re.match(regex, log) is not None

        if is_match(template1, template2):
            return template1
        if is_match(template2, template1):
            return template2
        return None

    def _insert_template(self, event_template, indexes):
        template_tokens = split_template_naive(event_template)
        self.template_items[event_template] = {
            'indexes': indexes,
        }
        self.template_list.append(event_template)

    def _update_template(self, new_template, new_indexes, idx):
        old_template = self.template_list[idx]
        template_tokens = split_template_naive(new_template)

        insert_indexes = self.template_items[old_template].get('indexes',
                                                               {}).copy()
        #print("insert_indexes: {}".format(insert_indexes))
        #print("new_indexes: {}".format(new_indexes))
        for k, v in new_indexes.items():
            if k in insert_indexes:
                insert_indexes[k] = merge_sorted_lists(v, insert_indexes[k])
            else:
                insert_indexes[k] = v
        self.template_items[new_template] = {
            'indexes': insert_indexes,
        }
        if new_template != old_template:
            self.template_items.pop(old_template)
            self.template_list.pop(idx)
            self.template_list.append(new_template)
        #print('#' * 20)
        #print(self.template_items)
        return insert_indexes

    def update_indexes(self, template, new_indexes):
        """
        Update the indexes of an existing template in the database.

        :param template: The log template whose indexes need to be updated.
        :param new_indexes: A dictionary of new indexes.
        """
        # old_template = self.template_list[idx]
        if template not in self.template_items:
            template_tokens = split_template_naive(template)
            self.template_items[template] = {
                'indexes': new_indexes,
            }
            self.template_list.append(template)
            return new_indexes
        else:
            indexes2 = self.template_items[template].get('indexes', {}).copy()
            for k, v in new_indexes.items():
                if k in indexes2:
                    indexes2[k] = merge_sorted_lists(v, indexes2[k])
                else:
                    indexes2[k] = v
            print(
                f"[TemplateDB] Update Indexes: {sum(len(v) for v in self.template_items[template].get('indexes', {}).values())} -> {sum(len(v) for v in indexes2.values())} for `{template}`"
            )
            self.template_items[template]['indexes'] = indexes2
            return indexes2
