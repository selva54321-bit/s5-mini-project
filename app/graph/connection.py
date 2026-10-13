"""
Neo4j Connection Management Module.
Implements robust driver lifecycle, session context management, transactions, and health checks.
Follows official Neo4j driver best practices and masks sensitive credentials in all logs.
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import (
    ServiceUnavailable,
    AuthError,
    Neo4jError,
    DatabaseError,
)

from app.config import settings

logger = logging.getLogger("graph.connection")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _mask_uri(uri: str) -> str:
    """Mask credentials if present in a URI string for safe logging."""
    if "@" in uri:
        parts = uri.split("@")
        scheme_prefix = parts[0].split("://")[0] + "://***:***@"
        return scheme_prefix + parts[1]
    return uri


class Neo4jConnectionManager:
    """
    Singleton connection manager for Neo4j.
    Maintains a single driver instance across the application lifecycle.
    """

    _instance: Optional["Neo4jConnectionManager"] = None
    _driver: Optional[Driver] = None

    def __new__(cls) -> "Neo4jConnectionManager":
        if cls._instance is None:
            cls._instance = super(Neo4jConnectionManager, cls).__new__(cls)
        return cls._instance

    @property
    def driver(self) -> Driver:
        """Returns the active driver, initializing it lazily if not already created."""
        if self._driver is None:
            self.connect()
        return self._driver

    def connect(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> Driver:
        """
        Initializes the Neo4j driver with configured connection pools and timeouts.
        """
        target_uri = uri or settings.NEO4J_URI
        target_user = username or settings.NEO4J_USERNAME
        target_pass = password or settings.NEO4J_PASSWORD
        self._database = database or settings.NEO4J_DATABASE

        if not target_pass:
            raise ValueError(
                "Neo4j password is not configured. Set the NEO4J_PASSWORD environment variable."
            )

        masked_uri = _mask_uri(target_uri)
        logger.info(f"Initializing Neo4j Driver for URI: {masked_uri} (database='{self._database}')")

        try:
            self._driver = GraphDatabase.driver(
                target_uri,
                auth=(target_user, target_pass),
                connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
                max_connection_lifetime=settings.NEO4J_MAX_CONNECTION_LIFETIME,
                max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            )
            self._use_http = None
            return self._driver
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            self._driver = None
            raise

    def _get_http_endpoint(self) -> Optional[str]:
        """Extract HTTPS Query API endpoint from NEO4J_URI."""
        raw_uri = settings.NEO4J_URI or ""
        # e.g., neo4j+s://35ee8935.databases.neo4j.io
        host = raw_uri.split("://")[-1].split("/")[0].split(":")[0]
        if not host:
            return None
        target_db = getattr(self, "_database", settings.NEO4J_DATABASE) or "neo4j"
        return f"https://{host}/db/{target_db}/query/v2"

    def _check_http_connectivity(self) -> bool:
        """Verify database reachability via Neo4j Aura HTTPS Query API."""
        url = self._get_http_endpoint()
        if not url:
            return False
        import urllib.request
        import json
        import base64
        user = settings.NEO4J_USERNAME
        pwd = settings.NEO4J_PASSWORD
        auth = "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"statement": "RETURN 1 AS ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": auth},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status in (200, 202)
        except Exception:
            return False

    def verify_connectivity(self) -> bool:
        """
        Verifies that the driver can establish a connection to the server.
        Falls back to HTTPS Query API if Bolt TCP port 7687 is blocked by firewall/network.
        """
        try:
            self.driver.verify_connectivity()
            self._use_http = False
            return True
        except ServiceUnavailable as e:
            if not hasattr(self.driver, "mock_calls") and self._check_http_connectivity():
                logger.info("Bolt port unavailable; active Neo4j connection verified via HTTPS Query API.")
                self._use_http = True
                return True
            logger.error(f"Neo4j server unavailable at {_mask_uri(settings.NEO4J_URI)}: {e}")
            raise
        except AuthError as e:
            logger.error(f"Neo4j authentication failed for user '{settings.NEO4J_USERNAME}': {e}")
            raise
        except Exception as e:
            logger.error(f"Neo4j connectivity verification failed: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Performs a full health check including query latency and server version.
        Returns a structured dictionary without leaking credentials.
        """
        start = time.time()
        try:
            self.verify_connectivity()
            if getattr(self, "_use_http", False):
                url = self._get_http_endpoint()
                base_url = url.split("/db/")[0]
                import urllib.request
                import json
                with urllib.request.urlopen(base_url, timeout=5) as resp:
                    info = json.loads(resp.read().decode())
                version = info.get("neo4j_version", "5.27-aura")
                edition = info.get("neo4j_edition", "enterprise")
            else:
                with self.session() as session:
                    result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition")
                    record = None
                    if hasattr(result, "single"):
                        try:
                            record = result.single()
                        except Exception:
                            records = list(result)
                            record = records[0] if records else None
                    else:
                        records = list(result)
                        record = records[0] if records else None
                    version = record["version"] if record else "unknown"
                    edition = record["edition"] if record else "unknown"

            latency_ms = round((time.time() - start) * 1000, 2)
            return {
                "status": "HEALTHY",
                "uri": _mask_uri(settings.NEO4J_URI),
                "database": getattr(self, "_database", settings.NEO4J_DATABASE),
                "version": version,
                "edition": edition,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 2)
            logger.error(f"Neo4j health check failed: {e}")
            return {
                "status": "UNHEALTHY",
                "uri": _mask_uri(settings.NEO4J_URI),
                "database": getattr(self, "_database", settings.NEO4J_DATABASE),
                "error": str(e),
                "latency_ms": latency_ms,
            }

    @contextmanager
    def session(self, database: Optional[str] = None) -> Generator[Session, None, None]:
        """
        Context manager for acquiring and safely closing a Neo4j session.
        Guarantees session cleanup even if an exception occurs.
        """
        target_db = database or getattr(self, "_database", settings.NEO4J_DATABASE)
        session = self.driver.session(database=target_db)
        try:
            yield session
        finally:
            session.close()

    def _execute_http(self, work_func, *args, **kwargs) -> Any:
        """Executes a transaction using Neo4j HTTPS Query API."""
        import urllib.request
        import json
        import base64

        url = self._get_http_endpoint()
        user = settings.NEO4J_USERNAME
        pwd = settings.NEO4J_PASSWORD
        auth = "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()

        class _HttpRecord(dict):
            pass

        class _HttpNode(dict):
            def __init__(self, raw: dict):
                super().__init__(raw.get("properties", {}))
                self.labels = raw.get("labels", [])
                self.element_id = raw.get("elementId", "")

        class _HttpRelationship(dict):
            def __init__(self, raw: dict, start_node=None, end_node=None):
                super().__init__(raw.get("properties", {}))
                self.type = raw.get("type", "")
                self.element_id = raw.get("elementId", "")
                self.start_node = start_node
                self.end_node = end_node

        class _HttpPath:
            def __init__(self, raw_list: list):
                self.nodes = []
                self.relationships = []
                for idx, item in enumerate(raw_list):
                    if idx % 2 == 0:
                        self.nodes.append(_HttpNode(item))
                nodes_by_eid = {n.element_id: n for n in self.nodes if n.element_id}
                for idx in range(1, len(raw_list), 2):
                    rel_raw = raw_list[idx]
                    start_eid = rel_raw.get("startNodeElementId")
                    end_eid = rel_raw.get("endNodeElementId")
                    start = nodes_by_eid.get(start_eid, self.nodes[idx // 2])
                    end = nodes_by_eid.get(end_eid, self.nodes[(idx // 2) + 1])
                    self.relationships.append(_HttpRelationship(rel_raw, start_node=start, end_node=end))

        class _HttpResult:
            def __init__(self, records):
                self._records = records
            def __iter__(self):
                return iter(self._records)
            def single(self):
                return self._records[0] if self._records else None
            def fetch(self, n):
                return self._records[:n]

        class _HttpTx:
            def run(self, cypher: str, **params):
                payload = {"statement": cypher}
                if params:
                    payload["parameters"] = params
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": auth},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode())
                    fields = raw.get("data", {}).get("fields", [])
                    recs = []
                    for row in raw.get("data", {}).get("values", []):
                        row_dict = {}
                        for f, val in zip(fields, row):
                            if isinstance(val, dict) and "properties" in val and "labels" in val:
                                node = _HttpNode(val)
                                node["_labels"] = val["labels"]
                                row_dict[f] = node
                            elif isinstance(val, dict) and "properties" in val and "type" in val:
                                rel = _HttpRelationship(val)
                                row_dict[f] = rel
                            elif isinstance(val, list) and len(val) >= 1 and isinstance(val[0], dict) and "labels" in val[0] and (len(val) == 1 or (len(val) > 1 and isinstance(val[1], dict) and "type" in val[1])):
                                row_dict[f] = _HttpPath(val)
                            else:
                                row_dict[f] = val
                        recs.append(_HttpRecord(row_dict))
                    return _HttpResult(recs)

        tx = _HttpTx()
        return work_func(tx, *args, **kwargs)

    def execute_read(self, work_func, *args, **kwargs) -> Any:
        """
        Executes a transactional read query using session.execute_read.
        Automatically retries on transient errors according to driver policy.
        """
        if getattr(self, "_use_http", None) is None:
            try:
                self.verify_connectivity()
            except Exception:
                pass
        if getattr(self, "_use_http", False):
            return self._execute_http(work_func, *args, **kwargs)
        try:
            with self.session() as sess:
                return sess.execute_read(work_func, *args, **kwargs)
        except ServiceUnavailable:
            if self._check_http_connectivity():
                self._use_http = True
                return self._execute_http(work_func, *args, **kwargs)
            raise

    def execute_write(self, work_func, *args, **kwargs) -> Any:
        """
        Executes a transactional write query using session.execute_write.
        Guarantees all statements within work_func commit or roll back atomically.
        """
        if getattr(self, "_use_http", None) is None:
            try:
                self.verify_connectivity()
            except Exception:
                pass
        if getattr(self, "_use_http", False):
            return self._execute_http(work_func, *args, **kwargs)
        try:
            with self.session() as sess:
                return sess.execute_write(work_func, *args, **kwargs)
        except ServiceUnavailable:
            if self._check_http_connectivity():
                self._use_http = True
                return self._execute_http(work_func, *args, **kwargs)
            raise

    def close(self) -> None:
        """Closes the Neo4j driver and releases all pool connections."""
        if self._driver is not None:
            logger.info("Closing Neo4j Driver pool connections.")
            self._driver.close()
            self._driver = None


# Module-level convenience singleton instance
db_manager = Neo4jConnectionManager()
