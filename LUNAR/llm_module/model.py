import re
import time
from openai import OpenAI
from LUNAR.llm_module.response_extractor.extract_batch import BatchExtract
from LUNAR.llm_module.post_process import post_process_template
from LUNAR.llm_module.template_aggregator import aggregate_by_majority
from LUNAR.llm_module.variable_examples import VARIABLE_EXAMPLES_SETTING, json2prompt


class InferLLMGrouping:

    def __init__(self,
                 model,
                 api_key,
                 base_url,
                 prefix=None,
                 dataset="Apache",
                 prompt="VarExam"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prefix = prefix
        self.dataset = dataset
        self.prompt = prompt

        self.module_params = {}
        self.messages = []
        self.usages = []
        self.response = ""

        self.system_prompt = (
            "You are a log parsing assistant for the cloud reliability team, "
            "skilled in identifying dynamic values of variables/parameters in logs. "
            "The value is an actual manifestation of the variables in original logging statements.\n"
        )
        self.prompt_base_requirements = (
            "# Basic Requirements:\n"
            "- I will provide multiple log messages, each delimited by backticks.\n"
            "- You must identify and extract all dynamic variables (some variables are identical across logs, but they are still variables. Please see variable types below) in each log with {placeholder} and output static log templates.\n"
            "- Identify the semantics of variables and compare the differences between logs to identify potential dynamic variables if they belong to the same template.\n"
            "- Preserve any dynamic variables already marked by `<*>` or `{placeholder}`.\n"
            #"- Pay attention to the slightly different strings among logs, which have high possibility to be dynamic variable.\n"
            "- Do not convert non-variables, especially when only one log is presented in the group.\n"
        )
        if "SimpleRequirements" in self.prompt:
            self.prompt_base_requirements = (
                "# Basic Requirements:\n"
                "- I want you to act like an expert in log parsing.\n"
                "- I will give you a log message wrapped by backticks.\n"
                "- Your task is to identify all the dynamic variables in logs, replace them with {variables}, and output a static log template.\n"
                "- Please print the input log's template wrapped by backticks.\n"
            )

        if "NoAdvice" not in self.prompt:
            self.prompt_variable_advice = (
                "# Advices on variables:\n"
                "- Common variables: numbers, IP addresses, **URLs**, file paths, directories, hex values, usernames, etc.\n"
                "- Full directory with filename, complex url with server address or domain should be recognize as one variable.\n"
                #"- A very long (of length 10 or greater) token (assuming the log is split by whitespace) is likely to be a variable.\n"
                "- All of the types listed above are variables, even if they are identical across multiple logs. **Please take the requirements seriously! If the substring (surrounded by whitespaces) falls into these types, mark them as {variable_type}, where variable_type is the corresponding type!**\n"
                "# Advices on non-variables:\n"
                "- Error messages/types, java exceptions, detailed commands or interrupted messages are NOT dynamic variables as they contain important information.\n"
                "- Specific actions or status words are NOT dynamic variables.\n"
                #"For patterns like `a=b`: `a` typically is NOT a variable, but `b` is.\n"
                #"However, if `b` is missing, i.e., only `a=`, then, do NOT write `a={variable}`.\n"
                #"Plz NOT use {variable} for representing an empty substring. E.g., `a=` should NOT be converted into `a={variable}`.\n"
                #"- Avoid labeling multiple words together as one variable.\n"
            )
        else:
            self.prompt_variable_advice = ""

        if "NoPE" not in self.prompt:
            self.prompt_variable_example_prompt = self.construct_variable_example(
            )
            print(self.prompt_variable_example_prompt)
        else:
            self.prompt_variable_example_prompt = ""

        if "NoOutputConstraint" not in self.prompt:
            self.prompt_output_constraint = (
                "# Output Constraints: \n"
                "- For each log line, output corresponding log template starting with LogTemplate[idx], no other line break. \n"
                "- Each input log's template is delimited by backticks. \n"
                "- Examples:\n"
                "  LogTemplate[1]: `this is log template 1`\n"
                "  LogTemplate[2]: `this is log template 2`\n")
        else:
            self.prompt_output_constraint = ""

        self.instruction = ""
        self.instruction += self.prompt_base_requirements
        self.instruction += self.prompt_variable_advice
        self.instruction += self.prompt_variable_example_prompt
        # lezhang.thu - start
        self.instruction += (
            "# NON-Variable Examples (these types of strings should NOT be replaced by {variaible_type}): \n"
            "- `java.io.FileNotFoundException`\n"
            "- `[auth]`\n"
            #"- `a` in `a=`, `a:` etc., i.e., strings preceding `=` or `:`\n"
            #"- those messages (e.g., `this is an error`) containing more than two words when splitting w.r.t. whitespace\n"
            #"- `pwd`, `ls`, `sshd` (Linux commands)\n"
            #"- `worker.jni:onShutdown` -> `{configuration_reference}`\n"
            #"- `HTTP/1.1` `HTTP/1.0` -> `{protocol_and_version}`\n"
        )
        # lezhang.thu - end
        self.instruction += self.prompt_output_constraint
        self.instruction += (
            #"AVOID labeling MULTIPLE CONSECUTIVE words TOGETHER as a SINGLE variable (e.g., error/interruption messages consisting of multiple words).\n"
            "Hint: For extracting the template, the first step is to replace the typical variable strings within the logs by {variable_type} before anything."
        )

        print("======================== Prompt ========================")
        print(self.prompt)
        print(self.instruction)
        print("======================== Prompt ========================")

    def construct_variable_example(self):
        pe_dict = VARIABLE_EXAMPLES_SETTING['lunar']['variable_examples']
        prompt = json2prompt(pe_dict)

        return prompt

    def get_prompt_direct(self,
                          logs,
                          exemplars=None,
                          prev_templates=None,
                          proposal=None):
        # print(instruction)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.instruction
            },
            {
                "role": "assistant",
                "content": "OK, I'm ready to help."
            },
        ]
        if exemplars is not None:
            examplar_logs = [exemplar['query'] for exemplar in exemplars]
            examplar_templates = [exemplar['answer'] for exemplar in exemplars]
            query_template = '\n'.join(
                [f"Log[{i+1}]: " + "`{}`" for i in range(len(exemplars))])
            answer_template = '\n'.join([
                f"LogTemplate[{i+1}]: " + "`{}`" for i in range(len(exemplars))
            ])
            messages.append({
                "role": "user",
                "content": query_template.format(*examplar_logs)
            })
            messages.append({
                "role":
                "assistant",
                "content":
                answer_template.format(*examplar_templates)
            })

        if len(prev_templates) > 0:
            wrong_alert = ''
            wrong_alert += (
                'Before parsing, we list several *faulty* templates:')
            for t_x in prev_templates:
                wrong_alert += '\n`{}`'.format(t_x)
            messages.append({"role": "user", "content": wrong_alert})
            print('wrong_alert:\n{}'.format(wrong_alert))
            messages.append({"role": "assistant", "content": "OK."})

        query_template = '\n'.join(
            [f"Log[{i+1}]: " + "`{}`" for i in range(len(logs))])
        query = query_template.format(*logs)
        if proposal:
            brain_proposal = (
                "\nLastly, the following sequence of words is likely derived from the template.\n"
                f"{proposal}")
                #"Try to locate the most common words across logs, and keep them as part of the template.")
            query += brain_proposal
        messages.append({"role": "user", "content": query})
        # self.messages = messages
        print("\t============  Query  ====================")
        print("\n".join(["\t" + i for i in query.split('\n')]))
        return messages

    def parsing_log_templates(self,
                              logs,
                              exemplars,
                              gts=[],
                              reparse=None,
                              proposal=None):
        # query llm for response
        messages = self.get_prompt_direct(logs,
                                          exemplars=exemplars,
                                          prev_templates=reparse,
                                          proposal=proposal)

        temperature = 0.7 if len(reparse) > 0 else 0.0
        time1 = time.time()
        _ = self.get_response_fallback(messages, temperature=temperature)
        query_time = time.time() - time1

        # print response
        print("\t============ Response ====================")
        print(self.response)
        if len(gts) > 0:
            print("\t============ Target ====================")
            answer_template = '\n'.join(
                [f"\tGT Template[{i+1}]: " + "`{}`" for i in range(len(gts))])
            print(answer_template.format(*gts))
        # print("================================")

        # post process response
        try:
            gpt_templates = self.extract_and_post_process(logs, self.response)
            templates = [temp['post_process'] for temp in gpt_templates]
        except:
            templates = [post_process_template(log, [])[0] for log in logs]
        # aggregate templates
        best_template = aggregate_by_majority(logs, templates)

        return best_template, query_time, gpt_templates[0]['template'], templates

    def match_log_pattern(self, template: str, log: str) -> bool:
        """
        Return True if the log matches the template pattern where
        '<*>' acts as a wildcard for any sequence of characters.
        If False, print the first place where template and log differ.
        """
        # Escape regex special characters
        regex = re.escape(template)
        # Replace '<\*>' (only * is escaped by re.escape) with wildcard
        regex = regex.replace(r'<\*>', '.*?')
        # Add anchors
        regex = '^' + regex + '$'
        match = re.match(regex, log)
        message = ""
        if match is not None:
            return True, message, regex

        # Find where it fails
        # Split template into parts: literals and wildcards
        parts = re.split(r'<\*>', template)

        pos = 0  # Current position in log
        for i, part in enumerate(parts):
            next_pos = log.find(part, pos)
            if i == 0 and next_pos != 0:
                message = (f"Mismatch: Expected log to start with '{part}'\n"
                           f"Log starts with: '{log[:min(50, len(log))]}...'")
                return False, message, regex
            elif next_pos == -1:
                message = (
                    f"Mismatch: Cannot find expected text '{part}' in log after position {pos}\n"
                    #f"Log before position {pos}: '{log[:pos]}' (which are successfully matched)\n"
                    f"Log from position {pos}: '{log[pos:min(pos+50, len(log))]}...'"
                )
                return False, message, regex
            pos = next_pos + len(part)
        # Check if there's extra content at the end
        if pos < len(log):
            message = (
                f"Mismatch: Extra content at end of log\n"
                f"Expected end at position {pos}, but log continues: '{log[pos:]}'"
            )
            return False, message, regex
        return False, message, regex

    def improve_template(self, logs, template):
        system_prompt = (
            "You are an assistant designed to refine a given template based on a set of logs. "
            "Your goal is to optimize the template so that it matches as many logs as possible."
        )
        code = (
            "    def match_log_pattern(self, template: str, log: str) -> bool:\n"
            "        \"\"\"\n"
            "        Return True if the log matches the template pattern where\n"
            "        '<*>' acts as a wildcard for any sequence of characters.\n"
            "        If False, print the first place where template and log differ.\n"
            "        \"\"\"\n"
            "        # Escape regex special characters\n"
            "        regex = re.escape(template)\n"
            "        # Replace '<\\*>' (only * is escaped by re.escape) with wildcard\n"
            "        regex = regex.replace(r'<\\*>', '.*?')\n"
            "        # Add anchors\n"
            "        regex = '^' + regex + '$'\n"
            "        match = re.match(regex, log)\n"
            "        message = \"\"\n"
            "        if match is not None:\n"
            "            return True, message, regex\n"
            "            \n"
            "        # Find where it fails\n"
            "        # Split template into parts: literals and wildcards\n"
            "        parts = re.split(r'<\\*>', template)\n"
            "        \n"
            "        pos = 0  # Current position in log\n"
            "        for i, part in enumerate(parts):\n"
            "            next_pos = log.find(part, pos)\n"
            "            if i == 0 and next_pos != 0:\n"
            "                message = (f\"Mismatch: Expected log to start with '{part}'\\n\"\n"
            "                           f\"Log starts with: '{log[:min(50, len(log))]}...'\")\n"
            "                return False, message, regex\n"
            "            elif next_pos == -1:\n"
            "                message = (\n"
            "                    f\"Mismatch: Cannot find expected text '{part}' in log after position {pos}\\n\"\n"
            #"                    f\"Log before position {pos}: '{log[:pos]}' (which are successfully matched)\\n\"\n"
            "                    f\"Log from position {pos}: '{log[pos:min(pos+50, len(log))]}...'\"\n"
            "                )\n"
            "                return False, message, regex\n"
            "            pos = next_pos + len(part)\n"
            "        # Check if there's extra content at the end\n"
            "        if pos < len(log):\n"
            "            message = (\n"
            "                f\"Mismatch: Extra content at end of log\\n\"\n"
            "                f\"Expected end at position {pos}, but log continues: '{log[pos:]}'\"\n"
            "            )\n"
            "            return False, message, regex\n"
            "        return False, message, regex\n")
        t_x = self.match_log_pattern(template, logs[0])
        error_message = t_x[1]
        regex = t_x[2]
        instruction = (
            "Symbols <*> in the given template serve as wildcards representing a contiguous sequence of characters.\n"
            "You should only use <*> for matching any substring of the log text.\n"
            "Other non-<*>characters between the template and the log should exactly correspond to each other.\n"
            #"The Python code used to check whether a template matches a log is shown below:\n"
            #"```python\n"
            #"import re\n"
            #"\n"
            #"def match_log_pattern(template: str, log: str) -> bool:\n"
            #"    # Escape regex special characters\n"
            #"    regex = re.escape(template)\n"
            #"    # Replace '<\\*>' (only * is escaped by re.escape) with wildcard\n"
            #"    regex = regex.replace(r'<\\*>', '.*?')\n"
            #"    # Add anchors\n"
            #"    regex = '^' + regex + '$'\n"
            #"    return re.match(regex, log) is not None"
            #"```\n"
            "At present, the given template fails to match any of the provided logs.\n"
            f"The code that performs the matching check is::\n{code}\n"
            #f"The translated regular expression corresponding to the template is:\n{regex}\n"
            f"The error message for the template matching the given Log[1] is:\n{error_message}\n"
            "Please present your updated template in the following format:\n"
            "ImprovedTemplate: `the updated template`\n"
            #"You must modify the template; submitting it unchanged is not allowed."
        )
        print('error_message:\n{}'.format(error_message))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": instruction,
            },
            {
                "role": "assistant",
                "content": "OK, I'm ready to help.",
            },
        ]

        query_template = '\n'.join(
            [f"Log[{i+1}]: " + "`{}`" for i in range(len(logs))])
        query = query_template.format(*logs)
        query += '\nTemplate: `{}`'.format(template)
        messages.append({"role": "user", "content": query})
        _ = self.get_response_fallback(messages, temperature=.1)
        print('#' * 30)
        print('Improving...')
        print(self.response)
        t = re.search(r"ImprovedTemplate:\s*`([^`]*)`", self.response)
        return t.group(1)

    def get_response(self, messages, temperature=0.0):
        answers = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            seed=1603,
            n=1,
            stop=None,
        )
        self.usages.append({
            "query": answers.usage.prompt_tokens,
            "response": answers.usage.completion_tokens
        })
        self.response = [
            response.message.content for response in answers.choices
            if response.finish_reason != 'length'
        ][0]
        return self.response

    def get_response_fallback(self, messages, temperature=0.0):
        retry_times = 0
        while retry_times < 3:
            try:
                answers = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    seed=1603,
                    n=1,
                    stop=None,
                )
                self.response = [
                    response.message.content for response in answers.choices
                ][0]
                return self.response

            except Exception as e:
                print("Exception :", e)
                if "list index out of range" in str(e):
                    break
                retry_times += 1
        return ""

    def get_compromise_response(self, logs):
        return post_process_template(logs[0], [])[0]

    def extract_and_post_process(self, logs, response):
        gpt_templates = BatchExtract.extract(response)

        # replace null template with previous template
        gpt_templates = self.make_up_template(logs, gpt_templates)

        print("\t============ PostProcess ====================")
        for temp in gpt_templates:
            new_temp, _ = post_process_template(temp['template'], [])
            temp['post_process'] = new_temp
        return gpt_templates

    @staticmethod
    def make_up_template(logs, templates):
        """
            replace missing template with previous template
        :param logs: a list of strings
        :param templates: a list of dictionaries, each dictionary contains 'idx' and 'template'
        :return:
        """
        templates = sorted(templates, key=lambda x: x['idx'])
        # remove null template
        templates = [d for d in templates if d.get('idx') != -1]
        if len(templates) == 0:
            return [{
                'idx': i + 1,
                'template': log
            } for i, log in enumerate(logs)]

        new_templates = []
        existing_idx = [d['idx'] for d in templates]
        # print(existing_idx)
        template_idx = -1
        for idx, _log in enumerate(logs):
            if idx + 1 not in existing_idx:
                new_templates.append({
                    'idx': idx + 1,
                    'template': templates[0]['template']
                })
            else:
                template_idx += 1
                new_templates.append({
                    'idx':
                    idx + 1,
                    'template':
                    templates[template_idx]['template']
                })
        new_templates = sorted(new_templates, key=lambda x: x['idx'])

        return new_templates
