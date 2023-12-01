from __future__ import annotations

import typing as t
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass

try:
    from llama_index.indices.query.embedding_utils import get_top_k_embeddings
    from llama_index.node_parser import SimpleNodeParser
    from llama_index.readers.schema import Document as LlamaindexDocument
    from llama_index.schema import BaseNode
except ImportError:
    raise ImportError(
        "llama_index must be installed to use this function. "
        "Please, install it with `pip install llama_index`."
    )

import numpy as np
import numpy.testing as npt
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document as LangchainDocument
from numpy.random import default_rng
from tqdm import tqdm

from ragas.llms import llm_factory
from ragas.testset.prompts import (
    ANSWER_FORMULATE,
    COMPRESS_QUESTION,
    CONDITIONAL_QUESTION,
    CONTEXT_FORMULATE,
    CONVERSATION_QUESTION,
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    REASONING_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
)
from ragas.testset.utils import load_as_score
from ragas.utils import load_as_json

if t.TYPE_CHECKING:
    from ragas.llms.base import RagasLLM


DEFAULT_TEST_DISTRIBUTION = {
    "simple": 0.4,
    "reasoning": 0.2,
    "multi_context": 0.2,
    "conditional": 0.2,
}

question_deep_map = {
    "reasoning": "_reasoning_question",
    "conditional": "_condition_question",
}

DataRow = namedtuple("DataRow", ["question", "ground_truth_context", "ground_truth", "question_type", "episode_done"])


@dataclass
class TestDataset:
    """
    TestDataset class
    """

    test_data: t.List[DataRow]

    def to_pandas(self) -> pd.DataFrame:
        data_samples = []
        for data in self.test_data:
            data = {
                    "question": data.question,
                    "ground_truth_context": data.ground_truth_context,
                    "ground_truth": data.ground_truth,
                    "question_type": data.question_type,
                    "episode_done": data.episode_done,
            }
            data_samples.append(data)

        return pd.DataFrame.from_records(data_samples)


class TestsetGenerator:

    """
    Ragas Test Set Generator

    Attributes
    ----------
    generator_llm: LangchainLLM
        LLM used for all the generator operations in the TestGeneration paradigm.
    critique_llm: LangchainLLM
        LLM used for all the filtering and scoring operations in TestGeneration
        paradigm.
    embeddings_model: Embeddings
        Embeddings used for vectorizing nodes when required.
    chat_qa: float
        Determines the fraction of conversational questions the resulting test set.
    chunk_size: int
        The chunk size of nodes created from data.
    test_distribution : dict
        Distribution of different types of questions to be generated from given
        set of documents. Defaults to {"easy":0.1, "reasoning":0.4, "conversation":0.5}
    """

    def __init__(
        self,
        generator_llm: RagasLLM,
        critic_llm: RagasLLM,
        embeddings_model: Embeddings,
        testset_distribution: t.Optional[t.Dict[str, float]] = None,
        chat_qa: float = 0.0,
        chunk_size: int = 1024,
        seed: int = 42,
    ) -> None:
        self.generator_llm = generator_llm
        self.critic_llm = critic_llm
        self.embedding_model = embeddings_model
        testset_distribution = testset_distribution or DEFAULT_TEST_DISTRIBUTION
        npt.assert_almost_equal(
            1,
            sum(testset_distribution.values()),
            err_msg="Sum of distribution should be 1",
        )

        probs = np.cumsum(list(testset_distribution.values()))
        types = testset_distribution.keys()
        self.testset_distribution = dict(zip(types, probs))

        self.chat_qa = chat_qa
        self.chunk_size = chunk_size
        self.threshold = 7.5
        self.rng = default_rng(seed)

    @classmethod
    def from_default(
        cls,
        openai_generator_llm: str = "gpt-3.5-turbo-16k",
        openai_filter_llm: str = "gpt-4",
        chat_qa: float = 0.3,
        chunk_size: int = 512,
        testset_distribution: dict = DEFAULT_TEST_DISTRIBUTION,
    ):
        generator_llm = llm_factory(openai_generator_llm)
        critic_llm = llm_factory(openai_filter_llm)
        embeddings_model = OpenAIEmbeddings()  # type: ignore
        return cls(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings_model=embeddings_model,
            chat_qa=chat_qa,
            chunk_size=chunk_size,
            testset_distribution=testset_distribution,
        )

    def _get_evolve_type(self) -> str:
        """
        Decides question evolution type based on probability
        """
        prob = self.rng.uniform(0, 1)
        return next(
            (
                key
                for key in self.testset_distribution.keys()
                if prob <= self.testset_distribution[key]
            ),
            "simple",
        )

    def _filter_context(self, context: str) -> bool:
        """
        context: str
            The input context

        Checks if the context is has information worthy of framing a question
        """
        human_prompt = SCORE_CONTEXT.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.critic_llm.generate(prompts=[prompt])
        output = results.generations[0][0].text.strip()
        score = load_as_score(output)
        return score >= self.threshold

    def _seed_question(self, context: str) -> str:
        human_prompt = SEED_QUESTION.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _filter_question(self, question: str) -> bool:
        human_prompt = FILTER_QUESTION.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        results = self.critic_llm.generate(prompts=[prompt])
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(self.generator_llm, results)
        return json_results.get("verdict") != "No"

    def _reasoning_question(self, question: str, context: str) -> str:
        return self._qc_template(REASONING_QUESTION, question, context)

    def _condition_question(self, question: str, context: str) -> str:
        return self._qc_template(CONDITIONAL_QUESTION, question, context)

    def _multicontext_question(
        self, question: str, context1: str, context2: str
    ) -> str:
        human_prompt = MULTICONTEXT_QUESTION.format(
            question=question, context1=context1, context2=context2
        )
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _compress_question(self, question: str) -> str:
        return self._question_transformation(COMPRESS_QUESTION, question=question)

    def _conversational_question(self, question: str) -> str:
        return self._question_transformation(CONVERSATION_QUESTION, question=question)

    def _question_transformation(self, prompt, question: str) -> str:
        human_prompt = prompt.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _qc_template(self, prompt, question, context) -> str:
        human_prompt = prompt.format(question=question, context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _generate_answer(self, question: str, context: t.List[str]) -> t.List[str]:
        return [
            self._qc_template(ANSWER_FORMULATE, qstn, context[i])
            for i, qstn in enumerate(question.split("\n"))
        ]

    def _generate_context(self, question: str, text_chunk: str) -> t.List[str]:
        return [
            self._qc_template(CONTEXT_FORMULATE, qstn, text_chunk)
            for qstn in question.split("\n")
        ]

    def _remove_nodes(
        self, available_indices: t.List[BaseNode], node_idx: t.List
    ) -> t.List[BaseNode]:
        for idx in node_idx:
            available_indices.remove(idx)
        return available_indices

    def _generate_doc_nodes_map(
        self, document_nodes: t.List[BaseNode]
    ) -> t.Dict[str, t.List[BaseNode]]:
        doc_nodes_map: t.Dict[str, t.List[BaseNode]] = defaultdict(list)
        for node in document_nodes:
            if node.ref_doc_id:
                doc_nodes_map[node.ref_doc_id].append(node)

        return doc_nodes_map  # type: ignore

    def _get_neighbour_node(
        self, node: BaseNode, related_nodes: t.List[BaseNode]
    ) -> t.List[BaseNode]:
        if len(related_nodes) < 2:
            warnings.warn("No neighbors exists")
            return [node]
        idx = related_nodes.index(node)
        ids = [idx - 1, idx] if idx == (len(related_nodes) - 1) else [idx, idx + 1]
        return [related_nodes[idx] for idx in ids]

    def _embed_nodes(self, nodes: t.List[BaseNode]) -> t.Dict[str, t.List[float]]:
        embeddings = {}
        for node in nodes:
            embeddings[node.id_] = list(
                self.embedding_model.embed_query(node.get_content())
            )

        return embeddings

    def generate(
        self,
        documents: t.List[LlamaindexDocument] | t.List[LangchainDocument],
        test_size: int,
    ) -> TestDataset:
        if not isinstance(documents[0], (LlamaindexDocument, LangchainDocument)):
            raise ValueError(
                "Testset Generatation only supports LlamaindexDocuments or LangchainDocuments"  # noqa
            )

        if isinstance(documents[0], LangchainDocument):
            # cast to LangchainDocument since its the only case here
            documents = t.cast(t.List[LangchainDocument], documents)
            documents = [
                LlamaindexDocument.from_langchain_format(doc) for doc in documents
            ]
        # Convert documents into nodes
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=0, include_metadata=True
        )
        documents = t.cast(t.List[LlamaindexDocument], documents)
        document_nodes: t.List[BaseNode] = node_parser.get_nodes_from_documents(
            documents=documents
        )
        # maximum 1 seed question per node
        if test_size > len(document_nodes):
            raise ValueError(
                """Maximum possible number of samples exceeded, 
                             reduce test_size or add more documents"""
            )

        available_nodes = document_nodes
        doc_nodes_map = self._generate_doc_nodes_map(document_nodes)
        count_neighbours = sum(len(val) > 1 for _, val in doc_nodes_map.items())
        if count_neighbours < len(documents) // 2:
            warnings.warn("Most documents are too short")

        count = 0
        samples = []

        pbar = tqdm(total=test_size)
        while count < test_size and available_nodes != []:
            evolve_type = self._get_evolve_type()
            curr_node = self.rng.choice(np.array(available_nodes), size=1)[0]
            available_nodes = self._remove_nodes(available_nodes, [curr_node])

            neighbor_nodes = doc_nodes_map[curr_node.source_node.node_id]

            # Append multiple nodes randomly to remove chunking bias
            size = self.rng.integers(1, 3)
            nodes = (
                self._get_neighbour_node(curr_node, neighbor_nodes)
                if size > 1 and evolve_type != "multi_context"
                else [curr_node]
            )

            text_chunk = " ".join([node.get_content() for node in nodes])
            score = self._filter_context(text_chunk)
            if not score:
                continue
            seed_question = self._seed_question(text_chunk)
            is_valid_question = self._filter_question(seed_question)
            if not is_valid_question:
                continue

            if evolve_type == "multi_context":
                # Find most similar chunk in same document
                node_embedding = self._embed_nodes([nodes[-1]])
                neighbor_nodes = self._remove_nodes(neighbor_nodes, nodes)
                neighbor_emb = self._embed_nodes(neighbor_nodes)

                _, indices = get_top_k_embeddings(
                    list(node_embedding.values())[0],
                    list(neighbor_emb.values()),
                    similarity_cutoff=self.threshold / 10,
                )
                if indices:
                    # type cast indices from list[Any] to list[int]
                    indices = t.cast(t.List[int], indices)
                    best_neighbor = neighbor_nodes[indices[0]]
                    question = self._multicontext_question(
                        question=seed_question,
                        context1=text_chunk,
                        context2=best_neighbor.get_content(),
                    )
                    text_chunk = "\n".join([text_chunk, best_neighbor.get_content()])
                else:
                    continue

            # for reasoning and conditional modes, evolve question with the
            # functions from question_deep_map
            else:
                evolve_fun = question_deep_map.get(evolve_type)
                question = (
                    getattr(self, evolve_fun)(seed_question, text_chunk)
                    if evolve_fun
                    else seed_question
                )

            # compress question or convert into conversational questions
            if evolve_type != "simple":
                prob = self.rng.uniform(0, 1)
                if self.chat_qa and prob <= self.chat_qa:
                    question = self._conversational_question(question=question)
                else:
                    question = self._compress_question(question=question)

            is_valid_question = self._filter_question(question)
            if is_valid_question:
                context = self._generate_context(question, text_chunk)
                is_conv = len(context) > 1
                answer = self._generate_answer(question, context)
                for i, (qstn, ctx, ans) in enumerate(zip(question.split("\n"), context, answer)):
                    episode_done = False if is_conv and i==0 else True
                    samples.append(
                            DataRow(qstn, [ctx], [ans], evolve_type, episode_done)
                        )
                count += 1
                pbar.update(count)

        return TestDataset(test_data=samples)
