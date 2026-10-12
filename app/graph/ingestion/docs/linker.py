"""
Cross-Domain Entity Resolver and Linker (Task 4).
Bridges document, architecture, and policy knowledge with the existing code graph.
Enforces strict disambiguation: ambiguous entities are flagged rather than arbitrarily guessed.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from app.graph.schema import RelationshipType, ProvenanceType
from app.graph.ingestion.code.resolver import ResolvedRelationship
from app.graph.ingestion.docs.parser import ParsedDocument
from app.graph.ingestion.docs.policy_extractor import ExtractedPolicy, ExtractedSecurityRule
from app.graph.ingestion.docs.architecture_extractor import ExtractedArchComponent


@dataclass
class AmbiguousMatch:
    query_name: str
    candidate_urns: List[str]
    source_doc_urn: str
    reason: str


@dataclass
class LinkerResult:
    relationships: List[ResolvedRelationship] = field(default_factory=list)
    ambiguous_matches: List[AmbiguousMatch] = field(default_factory=list)


class CrossDomainLinker:
    """
    Resolves targets for policy rules, architecture components, and documents against
    the active code and service entities in the knowledge graph.
    """

    def __init__(self, known_code_entities: Optional[Dict[str, List[str]]] = None):
        """
        known_code_entities: maps simple/normalized name -> list of entity URNs
        e.g. {"compliance-report-job": ["urn:service:k8s:compliance-report-job"], ...}
        """
        self.code_entities = known_code_entities or {}

    def link_all(
        self,
        docs: List[ParsedDocument],
        policies: List[ExtractedPolicy],
        components: List[ExtractedArchComponent],
        project_urn: str,
    ) -> LinkerResult:
        result = LinkerResult()

        # 1. Link Architecture Components -> Documents & Dependencies
        comp_map = {c.name.lower(): c.urn for c in components}
        comp_slug_map = {c.urn.split(":")[-1]: c.urn for c in components}

        for comp in components:
            # Component DOCUMENTED_BY Document
            result.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.DOCUMENTED_BY.value,
                    source_id=comp.urn,
                    target_id=comp.source_doc_urn,
                    properties={"page_or_slide": comp.page_or_slide},
                    provenance={
                        "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                        "confidence": comp.confidence,
                        "evidence_snippet": comp.evidence_snippet,
                    },
                )
            )

            # Project CONTAINS Component
            result.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.CONTAINS.value,
                    source_id=project_urn,
                    target_id=comp.urn,
                    properties={},
                    provenance={
                        "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                        "confidence": 1.0,
                    },
                )
            )

            # Component DEPENDS_ON another Component or known Service
            for dep_name in comp.dependencies:
                dep_clean = dep_name.lower().strip()
                target_urn = comp_map.get(dep_clean) or comp_slug_map.get(dep_clean)

                if not target_urn and dep_clean in self.code_entities:
                    candidates = self.code_entities[dep_clean]
                    if len(candidates) == 1:
                        target_urn = candidates[0]
                    else:
                        result.ambiguous_matches.append(
                            AmbiguousMatch(
                                query_name=dep_name,
                                candidate_urns=candidates,
                                source_doc_urn=comp.source_doc_urn,
                                reason="Multiple code entity candidates match dependency name",
                            )
                        )

                if target_urn:
                    result.relationships.append(
                        ResolvedRelationship(
                            rel_type=RelationshipType.DEPENDS_ON.value,
                            source_id=comp.urn,
                            target_id=target_urn,
                            properties={"dependency_type": "architectural"},
                            provenance={
                                "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                                "confidence": 0.95,
                                "evidence_snippet": comp.evidence_snippet,
                            },
                        )
                    )

        # 2. Link Policies -> Security Rules -> Targeted Components / Services
        for pol in policies:
            # Policy DOCUMENTED_BY Document
            result.relationships.append(
                ResolvedRelationship(
                    rel_type=RelationshipType.DOCUMENTED_BY.value,
                    source_id=pol.urn,
                    target_id=pol.source_doc_urn,
                    properties={},
                    provenance={
                        "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                        "confidence": pol.confidence,
                    },
                )
            )

            for rule in pol.rules:
                # Policy CONTAINS SecurityRule
                result.relationships.append(
                    ResolvedRelationship(
                        rel_type=RelationshipType.CONTAINS.value,
                        source_id=pol.urn,
                        target_id=rule.urn,
                        properties={"severity": rule.severity},
                        provenance={
                            "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                            "confidence": 1.0,
                            "evidence_snippet": rule.evidence_snippet,
                        },
                    )
                )

                # SecurityRule APPLIES_TO target scope / components
                for target_name in rule.target_scope:
                    norm_target = target_name.lower().strip()
                    matched_urn = None

                    # Check against architecture components
                    if norm_target in comp_map:
                        matched_urn = comp_map[norm_target]
                    elif norm_target in comp_slug_map:
                        matched_urn = comp_slug_map[norm_target]

                    # Check against known code entities / services
                    elif norm_target in self.code_entities:
                        candidates = self.code_entities[norm_target]
                        if len(candidates) == 1:
                            matched_urn = candidates[0]
                        else:
                            result.ambiguous_matches.append(
                                AmbiguousMatch(
                                    query_name=target_name,
                                    candidate_urns=candidates,
                                    source_doc_urn=rule.source_doc_urn,
                                    reason="Multiple code services match policy scope target",
                                )
                            )

                    if matched_urn:
                        # SecurityRule APPLIES_TO Entity
                        result.relationships.append(
                            ResolvedRelationship(
                                rel_type=RelationshipType.APPLIES_TO.value,
                                source_id=rule.urn,
                                target_id=matched_urn,
                                properties={"severity": rule.severity},
                                provenance={
                                    "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                                    "confidence": rule.confidence,
                                    "evidence_snippet": rule.evidence_snippet,
                                },
                            )
                        )
                        # Inverted: Entity GOVERNED_BY SecurityRule
                        result.relationships.append(
                            ResolvedRelationship(
                                rel_type=RelationshipType.GOVERNED_BY.value,
                                source_id=matched_urn,
                                target_id=rule.urn,
                                properties={"severity": rule.severity},
                                provenance={
                                    "provenance_type": ProvenanceType.DOC_EXPLICIT.value,
                                    "confidence": rule.confidence,
                                    "evidence_snippet": rule.evidence_snippet,
                                },
                            )
                        )

        return result
