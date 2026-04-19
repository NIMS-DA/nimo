import numpy as np
import csv
import json
import re
import anthropic


class LLMEP():
    """Class of LLMEP

    This class selects the next experimental candidates using a Large Language Model (LLM).
    It is a standalone class with no dependency on ai_tool_re.

    The user provides a Markdown prompt file that describes the experimental context.
    The candidates CSV and any observed data are injected automatically into the prompt.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals,
                 prompt_file, system_prompt_file, llm_model,
                 num_runs, log_file, api_key, max_tokens):
        """Constructor

        Args:
            input_file (str): the file for candidates (CSV with descriptor columns + objective columns)
            output_file (str): the file for proposals output
            num_objectives (int): the number of objective columns (rightmost columns in CSV)
            num_proposals (int): the number of proposals to select
            prompt_file (str): path to the Markdown file describing the experimental context
            system_prompt_file (str): path to a file containing the system prompt.
            llm_model (str): Anthropic model ID to use
            num_runs (int): number of independent LLM selection runs (for majority vote)
            log_file (str): if provided, save the full selection log as Markdown
            api_key (str): Anthropic API key. If None, reads from ANTHROPIC_API_KEY env variable.
            max_tokens (int): maximum number of tokens in the LLM response (default: 8192).

        """
        self.input_file         = input_file
        self.output_file        = output_file
        self.num_objectives     = num_objectives
        self.num_proposals      = num_proposals
        self.prompt_file        = prompt_file
        self.system_prompt_file = system_prompt_file
        self.model              = llm_model
        self.num_runs           = num_runs
        if num_runs == None: self.num_runs = 10
        self.log_file           = log_file
        if log_file == None: self.log_file = "log"
        self.client             = anthropic.Anthropic(api_key = api_key)
        self.max_tokens         = max_tokens
        if max_tokens == None: self.max_tokens = 8192


    # ------------------------------------------------------------------
    # Data loading (ported from RE.load_data)
    # ------------------------------------------------------------------

    def load_data(self):
        """Load candidates CSV and split into observed / unobserved rows.

        t_train is returned as a list of raw strings for compatibility with
        calc_ai's signature, but LLMEP does not use it internally.

        Returns:
            t_train (list[list[str]]): observed objective values (raw strings)
            X_all (list[list[str]]): all descriptor rows (raw strings; numeric, categorical, and text all accepted)
            train_actions (np.ndarray): indices of observed rows
            test_actions (list[int]): indices of unobserved rows
        """
        with open(self.input_file, 'r') as f:
            reader    = csv.reader(f)
            _headers  = next(reader)
            raw_rows  = list(reader)

        X_all         = []
        t_train       = []
        train_actions = []
        test_actions  = []

        for idx, row in enumerate(raw_rows):
            descriptors = row[:-self.num_objectives]  # kept as strings (numeric, categorical, text all accepted)
            objectives  = row[-self.num_objectives:]
            X_all.append(descriptors)
            # A row is "observed" if all objective cells are non-empty
            if all(v.strip() != '' for v in objectives):
                t_train.append(objectives)
                train_actions.append(idx)
            else:
                test_actions.append(idx)

        train_actions = np.array(train_actions)

        return t_train, X_all, train_actions, test_actions


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_prompt(self):
        """Load the user-supplied Markdown prompt file."""
        with open(self.prompt_file, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


    def _build_candidates_text(self, X_all, train_actions, test_actions):
        """Build a text table of all candidates for injection into the prompt.

        Descriptor values are kept as raw strings, supporting numeric, categorical, and free-text descriptors.
        Observed rows include their measured objective value; unobserved rows show 'unknown'.

        Returns:
            str: CSV-like text block
        """
        with open(self.input_file, 'r') as f:
            reader   = csv.reader(f)
            headers  = next(reader)
            raw_rows = list(reader)

        descriptor_headers = headers[:-self.num_objectives]
        objective_headers  = headers[-self.num_objectives:]

        lines = ["index," + ",".join(descriptor_headers) + "," + ",".join(objective_headers)]
        for idx, row in enumerate(raw_rows):
            descriptors    = row[:-self.num_objectives]
            objectives_str = [v if v.strip() != '' else 'unknown' for v in row[-self.num_objectives:]]
            lines.append(str(idx) + "," + ",".join(descriptors) + "," + ",".join(objectives_str))

        return "\n".join(lines), objective_headers


    def _build_messages(self, user_prompt_md, candidates_text, objective_headers):
        """Construct the system prompt and user message for the single LLM call."""
        with open(self.system_prompt_file, 'r', encoding='utf-8') as f:
            system = f.read()

        obj_cols = ", ".join(f"`{h}`" for h in objective_headers)
        is_are = 'is' if len(objective_headers) == 1 else 'are'

        user = (
            f"{user_prompt_md}\n"
            "\n---\n"
            "\n## Candidates\n"
            "\nThe following table lists all candidate experimental conditions.\n"
            f"- Rows with a value in {obj_cols} are already observed data points.\n"
            f"- Rows where {obj_cols} {is_are} 'unknown' have not yet been measured.\n"
            "\n```\n"
            f"{candidates_text}\n"
            "```\n"
            "\n---\n"
            "\n## Your Task\n"
            f"\nPerform {self.num_runs} independent selection runs, then aggregate by majority vote.\n"
            "\n### Step 1 Independent Runs\n"
            f"\nFor each run (Run 1 through Run {self.num_runs}):\n"
            "- Choose a **distinct exploration strategy**.\n"
            f"- Select exactly {self.num_proposals} indices from the 'index' column above.\n"
            f"- Only select rows where {obj_cols} {is_are} 'unknown'.\n"
            "- Document the strategy and the selected indices.\n"

            "\n### Step 2 Majority Vote\n"
            f"\nAfter all {self.num_runs} runs:\n"
            "- Count how many times each index was selected across all runs.\n"
            "- Rank indices by vote count (descending). Break ties by preferring lower index values.\n"
            f"- Select the top {self.num_proposals} indices as the final proposals.\n"
            "- Document the vote counts for all indices that received at least one vote.\n"
            "\n### Output Format\n"
            "\nAfter your reasoning, output **exactly one** JSON object in the following format.\n"
            "This must be the last thing in your response.\n"
            "\n```json\n"
            "{\n"
            '  "runs": [\n'
            '    {\n'
            '      "run": 1,\n'
            '      "strategy": "<one-line description>",\n'
            '      "selected_indices": [<index1>, <index2>, ...]\n'
            '    },\n'
            '    ...\n'
            '  ],\n'
            '  "vote_counts": {\n'
            '    "<index>": <count>,\n'
            '    ...\n'
            '  },\n'
            '  "final_selected_indices": [<index1>, <index2>, ...],\n'
            '  "reasons": {\n'
            '    "<index>": "<one-sentence reason why this composition was selected>",\n'
            '    ...\n'
            '  }\n'
            "}\n"
            "```\n"
        )
        return [{"role": "user", "content": user}], system


    def _parse_result(self, response_text):
        """Extract the result JSON from the LLM response.

        Raises:
            ValueError: if parsing fails
        """
        # Try last ```json ... ``` block first
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_blocks:
            try:
                return json.loads(json_blocks[-1])
            except json.JSONDecodeError:
                pass

        # Fallback: find the last {...} containing final_selected_indices
        for m in reversed(list(re.finditer(r'\{', response_text))):
            candidate = response_text[m.start():]
            depth, end = 0, -1
            for i, ch in enumerate(candidate):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                try:
                    data = json.loads(candidate[:end + 1])
                    if "final_selected_indices" in data:
                        return data
                except json.JSONDecodeError:
                    continue

        raise ValueError(
            f"Could not parse result JSON from LLM response:\n{response_text[:500]}"
        )


    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        """Select proposals by delegating all runs + majority vote to the LLM.

        Args:
            t_train (np.ndarray): observed objective values
            X_all (np.ndarray): all descriptor rows
            train_actions (np.ndarray): indices of observed rows
            test_actions (list[int]): indices of unobserved rows

        Returns:
            actions (list[int]): selected action indices (length = num_proposals)
        """
        user_prompt_md  = self._load_prompt()
        candidates_text, objective_headers = self._build_candidates_text(X_all, train_actions, test_actions)

        print(f"  [LLMEP] Sending request ({self.num_runs} runs + majority vote) ...")

        messages, system = self._build_messages(user_prompt_md, candidates_text, objective_headers)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )

        response_text = response.content[0].text
        result        = self._parse_result(response_text)

        # Validate: keep only indices that are actually unobserved
        test_actions_set = set(test_actions)
        actions = [int(i) for i in result["final_selected_indices"] if int(i) in test_actions_set]
        reasons = result.get("reasons", {})

        if len(actions) < self.num_proposals:
            print(f"  [LLMEP] Warning: only {len(actions)} valid indices returned "
                  f"(expected {self.num_proposals}). Padding with remaining candidates.")
            used = set(actions)
            for a in test_actions:
                if len(actions) >= self.num_proposals:
                    break
                if a not in used:
                    actions.append(a)

        actions = actions[:self.num_proposals]

        if self.log_file:
            log_lines = [
                "# LLMEP Selection Log\n",
                f"Model    : {self.model}",
                f"Runs     : {self.num_runs}",
                f"Proposals: {self.num_proposals}\n",
                "---\n",
                "## Full LLM Response\n",
                response_text,
                "\n---\n",
                "## Final Selected Indices (Python-validated)\n",
            ]
            for rank, action in enumerate(actions, 1):
                reason = reasons.get(str(action), "")
                log_lines.append(f"  {rank}. index={action}, descriptors={X_all[action]}, reason={reason}")
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(log_lines))
            print(f"  [LLMEP] Log saved to {self.log_file}")

        print(f"  [LLMEP] Final proposals: {actions}")
        return actions, reasons


    # ------------------------------------------------------------------
    # Entry point (ported from RE.select)
    # ------------------------------------------------------------------

    def select(self):
        """Run the full selection pipeline and write proposals CSV.

        Returns:
            str: "True" on success
        """
        print("Start selection of proposals by LLMEP!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions, reasons = self.calc_ai(
            t_train=t_train, X_all=X_all,
            train_actions=train_actions, test_actions=test_actions,
        )

        print('Proposals')

        with open(self.input_file, 'r') as f:
            indexes = f.readlines()[0].rstrip('\n').split(',')

        indexes       = ["actions"] + indexes[:-self.num_objectives]
        proposals_all = [indexes]

        for i, action in enumerate(actions):
            row = [str(action)] + [str(v) for v in X_all[action]]
            proposals_all.append(row)
            print("###")
            print("number =", i + 1)
            print("actions =", action)
            print("proposal =", X_all[action])
            print("###")

        with open(self.output_file, 'w', newline="") as f:
            csv.writer(f).writerows(proposals_all)

        print("Finish selection of proposals!")
        return "True"
