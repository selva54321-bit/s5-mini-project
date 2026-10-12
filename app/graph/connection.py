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
            return self._driver
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            self._driver = None
            raise

    def verify_connectivity(self) -> bool:
        """
        Verifies that the driver can establish a connection to the server.
        """
        try:
            self.driver.verify_connectivity()
            return True
        except ServiceUnavailable as e:
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
            with self.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] AS version, edition")
                record = result.single()
                version = record["version"] if record else "unknown"
                edition = record["edition"] if record else "unknown"

            latency_ms = round((time.time() - start) * 1000, 2)
            return {
                "status": "HEALTHY",
                "uri": _mask_uri(settings.NEO4J_URI),
                "database": self._database,
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

    def execute_read(self, work_func, *args, **kwargs) -> Any:
        """
        Executes a transactional read query using session.execute_read.
        Automatically retries on transient errors according to driver policy.
        """
        with self.session() as sess:
            return sess.execute_read(work_func, *args, **kwargs)

    def execute_write(self, work_func, *args, **kwargs) -> Any:
        """
        Executes a transactional write query using session.execute_write.
        Guarantees all statements within work_func commit or roll back atomically.
        """
        with self.session() as sess:
            return sess.execute_write(work_func, *args, **kwargs)

    def close(self) -> None:
        """Closes the Neo4j driver and releases all pool connections."""
        if self._driver is not None:
            logger.info("Closing Neo4j Driver pool connections.")
            self._driver.close()
            self._driver = None


# Module-level convenience singleton instance
db_manager = Neo4jConnectionManager()
