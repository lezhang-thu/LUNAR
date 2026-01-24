import numpy as np
import regex
from LUNAR.llm_module.template_merger import LLMTemplateMerger


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

    def __init__(self, model, api_key, base_url):
        self.template_items = {}
        self.tpl_llm = LLMTemplateMerger(model, api_key, base_url)

    def add_template(self, event_template, indexes={}, llm_logs=[]):
        # loop invariant: event_template = pass-in UNION empty set
        merged = False
        while len(self.template_items) > 0:
            event_tokens = split_template_naive(event_template)
            xyz = max(self.template_items,
                      key=lambda x: jaccard_similarity(
                          event_tokens,
                          split_template_naive(x),
                      ))
            new_template = self._judge_template_merge_combine(
                event_template,
                xyz,
                llm_logs,
                self.template_items[xyz]['llm_logs'],
            )
            if new_template is None:
                break
            else:
                indexes, llm_logs = self._merge(indexes, llm_logs, xyz)
                merged = True
                self.template_items.pop(xyz)
                print(f"[TemplateDB] Merge: `{event_template}` | `{xyz}`")
                print(f"[TemplateDB] Merged: -> `{new_template}`")
                event_template = new_template

        self.template_items[event_template] = {
            'indexes': indexes,
            'llm_logs': llm_logs,
        }
        if merged:
            return True, event_template, indexes
        else:
            return False, event_template, None

    def _judge_template_merge_combine(self, template1, template2, llm1, llm2):
        if template1 == template2:
            return template1
        import re

        def is_match(template, log):
            regex = re.escape(template)
            regex = regex.replace(r'<\*>', '.*?')
            regex = '^' + regex + '$'
            return re.match(regex, log) is not None

        if is_match(template1, template2) and (len(
                split_template_naive(template1)) == len(
                    split_template_naive(template2)) or self.tpl_llm.core(
                        template1,
                        template2,
                        llm1,
                        llm2,
                    )):
            return template1
        if is_match(template2, template1) and (len(
                split_template_naive(template2)) == len(
                    split_template_naive(template1)) or self.tpl_llm.core(
                        template2,
                        template1,
                        llm2,
                        llm1,
                    )):
            return template2
        return None

    def _merge(
        self,
        indexes,
        llm_logs,
        old_template,
    ):
        insert_indexes = self.template_items[old_template].get(
            'indexes').copy()
        insert_llm_logs = self.template_items[old_template].get('llm_logs')
        for k, v in indexes.items():
            if k in insert_indexes:
                insert_indexes[k] = merge_sorted_lists(v, insert_indexes[k])
            else:
                insert_indexes[k] = v
        return insert_indexes, (llm_logs + insert_llm_logs)[:5]

    def update_indexes(self, template, new_indexes):
        """
        Update the indexes of an existing template in the database.

        :param template: The log template whose indexes need to be updated.
        :param new_indexes: A dictionary of new indexes.
        """
        assert template in self.template_items
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
