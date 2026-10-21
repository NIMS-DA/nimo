"""NIMO LLMEP local revision based on old/ai_tool_llmep.py.
Web-search support retained; final vote aggregation is deterministic Python.
"""
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import time
import numpy as np
import csv
import json
import re
import anthropic


class _BaseLLMEP():
    """Class of LLMEP

    This class selects the next experimental candidates using a Large Language Model (LLM).
    It is a standalone class with no dependency on ai_tool_re.

    The user provides a Markdown prompt file that describes the experimental context.
    The candidates CSV and any observed data are injected automatically into the prompt.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals,
                 prompt_file, system_prompt_file, llm_model,
                 num_runs, log_file, api_key, max_tokens,
                 use_web_search=False,
                 web_search_tool_version="web_search_20250305",
                 max_web_searches=5):
        """Constructor

        Args:
            input_file (str): the file for candidates (CSV with descriptor columns + objective columns)
            output_file (str): the file for proposals output
            num_objectives (int): the number of objective columns (rightmost columns in CSV)
            num_proposals (int): the number of proposals to select
            prompt_file (str): path to the Markdown file describing the experimental context
            system_prompt_file (str): path to a file containing the system prompt.
            llm_model (str): Anthropic model ID to use
            num_runs (int): number of alternative proposals generated in one API response
            log_file (str): if provided, save the full selection log as Markdown
            api_key (str): Anthropic API key. If None, reads from ANTHROPIC_API_KEY env variable.
            max_tokens (int): maximum number of tokens in the LLM response (default: 8192).
            use_web_search (bool): if True, expose the Anthropic web_search server tool
                to the model so it can retrieve literature at inference time. If False
                (default), the model answers from parametric knowledge only.
            web_search_tool_version (str): tool type string for the web_search tool.
            max_web_searches (int): max number of searches the model may run per request.

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
        self.client             = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self.max_tokens         = max_tokens
        if max_tokens == None: self.max_tokens = 8192
        if type(self.num_runs) is not int or self.num_runs < 1:
            raise ValueError("num_runs must be a positive integer")
        if type(self.num_proposals) is not int or self.num_proposals < 1:
            raise ValueError("num_proposals must be a positive integer")
        if type(max_web_searches) is not int or max_web_searches < 1:
            raise ValueError("max_web_searches must be a positive integer")
        self.use_web_search          = use_web_search
        self.web_search_tool_version = web_search_tool_version
        self.max_web_searches        = max_web_searches


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
        system = Path(self.system_prompt_file).read_text(encoding="utf-8")
        user = (f"{user_prompt_md}\n\n## Candidates\n{candidates_text}\n\n"
                f"Generate exactly {self.num_runs} alternative proposals in this single response. "
                "These are alternative proposals, not independent API trials. "
                "Use a distinct exploration strategy for each proposal. "
                f"Each proposal must contain exactly {self.num_proposals} distinct integer indices "
                "from the index column, and only select rows with unknown observed labels. "
                "Do not count votes or select a final batch: Python will count one vote per "
                "proposal per index, sort by descending vote count, break ties by ascending "
                "integer index, and select the required number of points. "
                "Return a single JSON object and no other text, with this structure: "
                '{"runs":[{"run":1,"strategy":"brief explanation","selected_indices":[0,1]}]}. '
                "The example is schematic: use the required number of runs and indices. "
                "Number the runs consecutively from 1.")
        return [{"role":"user", "content":user}], system


    def _parse_result(self, response_text):
        def unique_keys(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("Duplicate JSON key")
                obj[key] = value
            return obj
        text = response_text.strip()
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        result = json.loads(text, object_pairs_hook=unique_keys)
        if not isinstance(result, dict):
            raise ValueError("Response must be a JSON object")
        return result

    def _aggregate_votes(self, result, test_actions):
        runs = result.get("runs")
        if not isinstance(runs, list) or len(runs) != self.num_runs:
            raise ValueError(f"Expected exactly {self.num_runs} proposals")
        available = set(map(int, test_actions))
        votes = Counter()
        for number, proposal in enumerate(runs, 1):
            if not isinstance(proposal, dict):
                raise ValueError(f"Proposal {number} is not an object")
            if type(proposal.get("run")) is not int or proposal["run"] != number:
                raise ValueError(f"Proposal {number} has an invalid run number")
            ids = proposal.get("selected_indices")
            if not isinstance(ids, list) or len(ids) != self.num_proposals:
                raise ValueError(f"Proposal {number} must contain {self.num_proposals} indices")
            if any(type(i) is not int for i in ids):
                raise ValueError(f"Proposal {number} has a non-integer index")
            if len(set(ids)) != self.num_proposals:
                raise ValueError(f"Proposal {number} repeats an index")
            if not set(ids) <= available:
                raise ValueError(f"Proposal {number} selects an observed or unknown index")
            votes.update(ids)
        selected = sorted(votes, key=lambda i: (-votes[i], i))[:self.num_proposals]
        if len(selected) != self.num_proposals:
            raise ValueError("Not enough valid candidates")
        return selected, dict(sorted(votes.items()))


    def _extract_text(self, response):
        """Concatenate all text blocks and collect web-search provenance.

        When the web_search tool is enabled, response.content is a mix of
        'text', 'server_tool_use', and 'web_search_tool_result' blocks, so the
        old `response.content[0].text` decode is no longer safe. This joins every
        text block and, for reproducibility / data-leakage analysis, also returns
        the cited sources and the number of searches actually performed.

        Returns:
            response_text (str): concatenated text of all text blocks
            citations (list[dict]): [{'url':..., 'title':...}, ...] from web search
            n_searches (int): number of web_search_requests actually executed
        """
        texts     = []
        citations = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)
                for c in (getattr(block, "citations", None) or []):
                    url = getattr(c, "url", None)
                    if url:
                        citations.append({"url": url, "title": getattr(c, "title", None)})
        response_text = "\n".join(texts)

        n_searches = 0
        usage = getattr(response, "usage", None)
        stu   = getattr(usage, "server_tool_use", None) if usage is not None else None
        if stu is not None:
            n_searches = getattr(stu, "web_search_requests", 0) or 0

        return response_text, citations, n_searches


    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        if len(test_actions) < self.num_proposals:
            raise ValueError("Not enough unobserved candidates")
        audit_path = Path(str(self.log_file) + ".json")
        if audit_path.exists() or Path(self.output_file).exists():
            raise FileExistsError("Use fresh output/log paths to avoid overwriting a previous attempt")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_text, headers = self._build_candidates_text(X_all, train_actions, test_actions)
        messages, system = self._build_messages(self._load_prompt(), candidates_text, headers)
        request = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
        if self.use_web_search:
            request["tools"] = [{"type":self.web_search_tool_version, "name":"web_search",
                                  "max_uses":self.max_web_searches}]
        audit = {"implementation":"python_vote_v1", "requested_model":self.model,
                 "num_runs":self.num_runs, "num_proposals":self.num_proposals,
                 "use_web_search":self.use_web_search, "request":request,
                 "started_utc":datetime.now(timezone.utc).isoformat(),
                 "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "status":"request_started"}
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        text = ""
        started = time.perf_counter()
        try:
            response = self.client.messages.create(**request)
            text, citations, searches = self._extract_text(response)
            audit.update(response=response.model_dump(mode="json"), citations=citations,
                         web_search_requests=searches, response_text=text)
            # Save even truncated, tool-paused, or malformed responses for diagnosis.
            audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
            if response.stop_reason != "end_turn":
                raise ValueError(f"Incomplete response: stop_reason={response.stop_reason}")
            result = self._parse_result(text)
            actions, votes = self._aggregate_votes(result, test_actions)
            audit.update(status="validated", python_vote_counts=votes, selected_indices=actions)
            reasons = {str(i):f"Selected by Python aggregation: {votes[i]} votes" for i in actions}
            return actions, reasons
        except Exception as exc:
            # Do not copy API error strings or headers into logs: they may contain secrets.
            audit.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, (ValueError, FileExistsError)):
                audit["validation_error"] = str(exc)
            raise
        finally:
            audit["elapsed_seconds"] = time.perf_counter() - started
            audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
            Path(self.log_file).write_text(
                "# LLMEP Python-vote log\n\n"
                + f"Status: {audit['status']}\nModel: {self.model}\nWeb search: {self.use_web_search}\n"
                + "\n## Full response\n" + text
                + "\n\n## Python selection\n" + json.dumps(audit.get("selected_indices", []))
                + "\n\n## Python votes\n" + json.dumps(audit.get("python_vote_counts", {}))
                + "\n\nSee the adjacent JSON for request, usage, citations and validation details.\n",
                encoding="utf-8")


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

"""Validated single-batch selection or Python voting over alternative proposals."""
import hashlib,json,random,re,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path


class LLMEP(_BaseLLMEP):
    def __init__(self,*args,tie_seed=0,**kwargs):
        super().__init__(*args,**kwargs)
        if type(tie_seed) is not int or tie_seed < 0:raise ValueError('tie_seed must be a nonnegative integer')
        self.tie_seed=tie_seed

    def _build_messages(self,user_prompt_md,candidates_text,objective_headers):
        system=Path(self.system_prompt_file).read_text()
        if self.num_runs==1:
            instruction=f'Propose one batch of exactly {self.num_proposals} compositions. Select them as a coherent batch.'
        else:
            instruction=(f'Generate exactly {self.num_runs} alternative proposals with distinct exploration strategies, '
                         f'each containing exactly {self.num_proposals} compositions. These are alternatives in one response, not independent API trials. '
                         'Python will count one vote per proposal per index, select by descending vote count, and resolve tied votes by a seeded uniform random permutation. Do not aggregate votes yourself.')
        common=('Each selected index must be an integer from the index column whose observed label is unknown. '
                'Do not repeat an index within a proposal. Return one JSON object: '
                '{"runs":[{"run":1,"strategy":"brief explanation","selected_indices":[...]}]}. '
                f'Include exactly {self.num_runs} runs numbered consecutively from 1. Each list must contain exactly {self.num_proposals} valid indices.')
        return [{'role':'user','content':f'{user_prompt_md}\n\n## Candidates\n{candidates_text}\n\n## Selection protocol\n{instruction}\n{common}'}],system

    def _parse_result(self,text):
        # A single fenced JSON object may be preceded by explanation; don't
        # conflate harmless formatting with invalid scientific proposals.
        blocks=re.findall(r'```(?:json)?\s*(.*?)\s*```',text,re.S)
        if len(blocks)>1:raise ValueError('Return exactly one JSON object, not multiple blocks')
        return super()._parse_result(blocks[0] if blocks else text)

    def _aggregate_votes(self,result,test_actions):
        available=set(map(int,test_actions))
        runs=result.get('runs')
        if isinstance(runs,list):
            for index,run in enumerate(runs,1):
                if not isinstance(run,dict):continue
                ids=run.get('selected_indices')
                if not isinstance(ids,list) or any(type(i)is not int for i in ids):continue
                invalid=sorted(set(ids)-available)
                if invalid:raise ValueError(f'Proposal {index} selected unavailable IDs {invalid}. Use only rows labelled unknown.')
                repeated=sorted(i for i,n in Counter(ids).items() if n>1)
                if repeated:raise ValueError(f'Proposal {index} repeats IDs {repeated}; all IDs in each proposal must differ.')
        _,votes=super()._aggregate_votes(result,test_actions)
        if self.num_runs==1:return list(result['runs'][0]['selected_indices']),votes
        rng=random.Random(self.tie_seed);ranked=[]
        for count in sorted(set(votes.values()),reverse=True):
            group=sorted(i for i,v in votes.items() if v==count);rng.shuffle(group);ranked.extend(group)
        return ranked[:self.num_proposals],votes

    def calc_ai(self,t_train,X_all,train_actions,test_actions):
        if len(test_actions)<self.num_proposals:raise ValueError('Insufficient candidates')
        summary_path=Path(str(self.log_file)+'.json')
        if summary_path.exists() or Path(self.output_file).exists():raise FileExistsError('Use fresh output paths')
        candidates,headers=self._build_candidates_text(X_all,train_actions,test_actions)
        messages,system=self._build_messages(self._load_prompt(),candidates,headers)
        summary={'protocol':'validated_llmep_python_voting','num_runs':self.num_runs,'num_proposals':self.num_proposals,
                 'tie_seed':self.tie_seed,'use_web_search':self.use_web_search,'model':self.model,'max_attempts':3,
                 'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started','attempts':[],
                 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        summary_path.write_text(json.dumps(summary,indent=2))
        for attempt in range(1,4):
            request={'model':self.model,'max_tokens':self.max_tokens,'system':system,'messages':list(messages)}
            if self.use_web_search:
                request['tools']=[{'type':self.web_search_tool_version,'name':'web_search','max_uses':self.max_web_searches}]
            record={'attempt':attempt,'request':request,'status':'request_started'}
            path=Path(str(self.log_file)+f'.attempt_{attempt}.json');path.write_text(json.dumps(record,indent=2))
            start=time.perf_counter()
            text=''
            try:
                response=self.client.messages.create(**request)
                text,citations,searches=self._extract_text(response)
                record.update(response=response.model_dump(mode='json'),response_text=text,citations=citations,web_search_requests=searches)
                if response.stop_reason!='end_turn':raise ValueError(f'Incomplete response: {response.stop_reason}. Return a complete concise JSON object.')
                result=self._parse_result(text);ids,votes=self._aggregate_votes(result,test_actions)
                record.update(status='validated',selected_indices=ids,python_vote_counts=votes)
            except ValueError as exc:
                record.update(status='invalid',validation_error=str(exc))
            except Exception as exc:
                record.update(status='api_error',error_type=type(exc).__name__)
                raise
            finally:
                record['elapsed_seconds']=time.perf_counter()-start
                path.write_text(json.dumps(record,indent=2))
                summary['attempts'].append(str(path));summary['status']=record['status'];summary_path.write_text(json.dumps(summary,indent=2))
            if record['status']=='validated':
                summary.update(status='validated',selected_indices=ids,python_vote_counts=votes,accepted_attempt=attempt)
                summary_path.write_text(json.dumps(summary,indent=2))
                Path(self.log_file).write_text('# Validated selection\n\n'+json.dumps(summary,indent=2))
                return ids,{str(i):f'{votes[i]} votes' for i in ids}
            if attempt<3:
                messages=messages+[{'role':'assistant','content':text},{'role':'user','content':
                    'Your previous response failed validation: '+record['validation_error']+
                    f' Return a corrected complete JSON object with all {self.num_runs} proposals. '
                    'Do not select any observed point. The observations and scientific objective are unchanged. '
                    'Allowed unobserved IDs: '+json.dumps(list(map(int,test_actions)))}]
        raise ValueError('Three attempts exhausted; no batch adopted')
