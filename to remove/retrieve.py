"""Retrieval of protein-protein interaction data from public databases.

This module is the only place in the package that talks to the network.
Everything downstream consumes the normalised :class:`pandas.DataFrame`
returned by :meth:`InteractionSource.fetch_interactions`, so swapping
STRING for BioGRID (or for a local TSV) changes nothing else.

Normalised edge schema
----------------------
protein_a, protein_b : str   database-native protein identifiers
gene_a, gene_b       : str   human-readable gene symbols
score                : float confidence in [0, 1]
source               : str   name of the originating database
<extras>              : any   database-specific columns, kept as-is
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

EDGE_COLUMNS = ["protein_a", "protein_b", "gene_a", "gene_b", "score", "source"]


class RetrievalError(RuntimeError):
    """Raised when a database query fails or returns nothing usable."""


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


@dataclass
class ResponseCache:
    """On-disk cache of raw HTTP response bodies.

    Caching the *raw* body rather than the parsed frame means a bug in the
    parser can be fixed without re-hitting the API, and it makes test runs
    fully offline once the fixtures exist.
    """

    directory: Path = Path(".ppinet_cache")
    enabled: bool = True

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(url: str, params: dict[str, Any]) -> str:
        payload = json.dumps({"url": url, "params": params}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.txt"
        if path.exists():
            log.debug("cache hit %s", key)
            return path.read_text(encoding="utf-8")
        return None

    def put(self, key: str, body: str) -> None:
        if self.enabled:
            (self.directory / f"{key}.txt").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Abstract source
# --------------------------------------------------------------------------- #


class InteractionSource(ABC):
    """Common interface every PPI database adapter implements."""

    name: str

    @abstractmethod
    def map_identifiers(self, identifiers: Sequence[str]) -> pd.DataFrame:
        """Resolve free-text gene names to database-native identifiers.

        Returns a frame with at least ``query``, ``resolved_id`` and
        ``preferred_name``. Unresolvable queries appear with NA so callers
        can report the mapping rate instead of silently losing input.
        """

    @abstractmethod
    def fetch_interactions(self, identifiers: Sequence[str], **kwargs) -> pd.DataFrame:
        """Return edges in the normalised schema documented at module level."""


# --------------------------------------------------------------------------- #
# STRING
# --------------------------------------------------------------------------- #


@dataclass
class StringDBSource(InteractionSource):
    """Client for the STRING REST API.

    Parameters
    ----------
    species
        NCBI taxonomy identifier (9606 = *Homo sapiens*).
    required_score
        Minimum STRING combined score, 0-1000. STRING's own conventions:
        150 = low, 400 = medium, 700 = high, 900 = highest. This single
        number changes the topology of everything downstream, so it is a
        deliberate parameter and never a hidden default.
    network_type
        ``"functional"`` (all evidence channels) or ``"physical"``
        (physical complex membership only). Say which one you used in the
        report -- they answer different biological questions.
    caller_identity
        STRING asks API users to identify themselves. Put your project URL
        or email here.
    base_url
        Pin this to a versioned host for reproducible results. The
        unversioned host tracks whatever release is current.
    """

    species: int = 9606
    required_score: int = 400
    network_type: str = "functional"
    caller_identity: str = "ppinet-scientific-programming-project"
    base_url: str = "https://version-12-0.string-db.org/api"
    cache: ResponseCache = field(default_factory=ResponseCache)
    min_interval: float = 1.0  # STRING asks for <= 1 request/second
    timeout: int = 60

    name: str = field(default="STRING", init=False)
    _last_call: float = field(default=0.0, init=False, repr=False)
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.required_score <= 1000:
            raise ValueError("required_score must be in [0, 1000]")
        if self.network_type not in {"functional", "physical"}:
            raise ValueError("network_type must be 'functional' or 'physical'")
        self._session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    # -- low level ---------------------------------------------------------- #

    def _request(self, method: str, params: dict[str, Any], fmt: str = "tsv") -> str:
        """POST to ``{base_url}/{fmt}/{method}``, honouring cache and throttle.

        POST rather than GET because URLs have a length limit and a few
        hundred identifiers overflow it.
        """
        url = f"{self.base_url}/{fmt}/{method}"
        params = {**params, "caller_identity": self.caller_identity}

        cache_key = ResponseCache.key(url, params)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        log.info("STRING %s (%d identifiers)", method, len(params.get("identifiers", "").split("\r")))
        response = self._session.post(url, data=params, timeout=self.timeout)
        self._last_call = time.monotonic()

        if response.status_code == 404:
            raise RetrievalError(
                f"STRING resolved none of the submitted identifiers for species "
                f"{self.species}. Check the species id and the identifier type."
            )
        response.raise_for_status()

        body = response.text
        if not body.strip():
            raise RetrievalError(f"STRING returned an empty body for method '{method}'.")
        self.cache.put(cache_key, body)
        return body

    @staticmethod
    def _encode(identifiers: Iterable[str]) -> str:
        """STRING separates multiple identifiers with a carriage return."""
        return "\r".join(identifiers)

    # -- public ------------------------------------------------------------- #

    def map_identifiers(self, identifiers: Sequence[str], limit: int = 1) -> pd.DataFrame:
        """Resolve gene symbols / UniProt accessions to STRING protein ids.

        Always do this before calling :meth:`fetch_interactions`. STRING will
        accept raw symbols, but pre-resolved ids (``9606.ENSP00000269305``)
        disambiguate synonyms and are answered much faster.
        """
        body = self._request(
            "get_string_ids",
            {
                "identifiers": self._encode(identifiers),
                "species": self.species,
                "limit": limit,
                "echo_query": 1,
            },
        )
        raw = pd.read_csv(StringIO(body), sep="\t")

        mapping = raw.rename(
            columns={
                "queryItem": "query",
                "stringId": "resolved_id",
                "preferredName": "preferred_name",
            }
        )[["query", "resolved_id", "preferred_name", "annotation"]]

        # Re-attach anything STRING dropped, so the caller sees the failures.
        missing = set(identifiers) - set(mapping["query"])
        if missing:
            log.warning("%d/%d identifiers unresolved: %s",
                        len(missing), len(identifiers), sorted(missing))
            mapping = pd.concat(
                [mapping, pd.DataFrame({"query": sorted(missing)})], ignore_index=True
            )
        return mapping

    def fetch_interactions(
        self,
        identifiers: Sequence[str],
        add_nodes: int = 0,
        resolve_first: bool = True,
    ) -> pd.DataFrame:
        """Fetch the interaction network induced by ``identifiers``.

        Parameters
        ----------
        add_nodes
            Ask STRING to pad the network with its N highest-confidence
            interactors outside the query set. Useful for hub discovery,
            but it biases enrichment: the added nodes were chosen *because*
            they are well connected. Keep it at 0 for the primary analysis
            and treat any non-zero run as a separate, clearly labelled one.
        resolve_first
            Map identifiers via :meth:`map_identifiers` before querying.
        """
        if resolve_first:
            mapping = self.map_identifiers(identifiers)
            query_ids = mapping["resolved_id"].dropna().tolist()
            if not query_ids:
                raise RetrievalError("No identifiers could be resolved against STRING.")
        else:
            query_ids = list(identifiers)

        body = self._request(
            "network",
            {
                "identifiers": self._encode(query_ids),
                "species": self.species,
                "required_score": self.required_score,
                "network_type": self.network_type,
                "add_nodes": add_nodes,
            },
        )
        raw = pd.read_csv(StringIO(body), sep="\t")
        return self._normalise(raw)

    def _normalise(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map STRING's columns onto the package-wide edge schema.

        STRING returns per-channel evidence subscores (nscore, fscore,
        pscore, ascore, escore, dscore, tscore) alongside the combined
        score. Keep them: they are exactly the edge metadata the brief
        asks you to preserve, and they let you filter to, say, edges with
        experimental support only.
        """
        edges = pd.DataFrame(
            {
                "protein_a": raw["stringId_A"],
                "protein_b": raw["stringId_B"],
                "gene_a": raw["preferredName_A"],
                "gene_b": raw["preferredName_B"],
                "score": raw["score"].astype(float),
                "source": self.name,
            }
        )
        channels = ["nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore"]
        for channel in channels:
            if channel in raw.columns:
                edges[channel] = raw[channel].astype(float)

        # STRING lists each undirected pair once, but be defensive: an
        # unordered-key dedup protects against A-B / B-A duplicates that
        # would otherwise inflate degree.
        key = edges[["protein_a", "protein_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
        edges = edges.loc[~key.duplicated()].reset_index(drop=True)
        return edges

    def version(self) -> str:
        """Report the live STRING release -- worth logging into every run."""
        return self._request("version", {}, fmt="tsv-no-header").strip()


# --------------------------------------------------------------------------- #
# Local files, for tests and offline demos
# --------------------------------------------------------------------------- #


@dataclass
class LocalFileSource(InteractionSource):
    """Read a previously saved edge table. Same interface, no network."""

    path: Path
    name: str = field(default="local", init=False)

    def map_identifiers(self, identifiers: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame({"query": list(identifiers),
                             "resolved_id": list(identifiers),
                             "preferred_name": list(identifiers)})

    def fetch_interactions(self, identifiers: Sequence[str], **kwargs) -> pd.DataFrame:
        edges = pd.read_csv(self.path, sep=None, engine="python")
        missing = set(EDGE_COLUMNS) - set(edges.columns)
        if missing:
            raise RetrievalError(f"{self.path} is missing columns: {sorted(missing)}")
        if identifiers:
            wanted = set(identifiers)
            edges = edges[edges["gene_a"].isin(wanted) & edges["gene_b"].isin(wanted)]
        return edges.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# BioGRID -- stub for the second adapter
# --------------------------------------------------------------------------- #


@dataclass
class BioGridSource(InteractionSource):
    """BioGRID REST client. Requires a free access key from the BioGRID site.

    Left as a stub deliberately: implement it only once STRING works end to
    end. The value of the second source is showing that the abstraction
    holds, so the only thing that should change downstream is which object
    you instantiate.
    """

    access_key: str
    taxon_id: int = 9606
    base_url: str = "https://webservice.thebiogrid.org"
    name: str = field(default="BioGRID", init=False)

    def map_identifiers(self, identifiers: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_interactions(self, identifiers: Sequence[str], **kwargs) -> pd.DataFrame:
        raise NotImplementedError


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    seeds = ["TP53", "MDM2", "CDKN1A", "ATM", "CHEK2", "BAX", "BBC3", "CDK2"]

    string = StringDBSource(species=9606, required_score=700, network_type="physical")
    print("STRING release:", string.version())

    mapping = string.map_identifiers(seeds)
    print(mapping[["query", "resolved_id", "preferred_name"]].to_string(index=False))

    edges = string.fetch_interactions(seeds)
    print(f"\n{len(edges)} edges among {len(seeds)} seed proteins")
    print(edges[EDGE_COLUMNS].head(10).to_string(index=False))

    edges.to_csv("data/example/tp53_edges.tsv", sep="\t", index=False)
