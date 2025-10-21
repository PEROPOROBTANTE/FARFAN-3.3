# coding=utf-8
"""
Deterministic Sample Data Generator
===================================

SIN_CARRETA: Generate deterministic sample data using seeded RNG for all API endpoints.
Ensures reproducibility and contract compliance.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from api.utils.seeded_rng import create_seeded_generator
from api.models.schemas import (
    RegionSummary, RegionDetail, RegionCoordinates, RegionMetadata,
    MunicipalitySummary, MunicipalityDetail, MunicipalityMetadata,
    QuestionAnalysis, DimensionAnalysis, Evidence,
    Cluster, ClusterMember,
    QualitativeLevelEnum, DimensionEnum, PolicyAreaEnum,
    # Visualization schemas
    ConstellationNode, ConstellationEdge, ConstellationVisualizationResponse,
    PhylogramNode, PhylogramVisualizationResponse,
    MeshNode, MeshVisualizationResponse,
    HelixPoint, HelixVisualizationResponse,
    RadarAxis, RadarVisualizationResponse,
    # Temporal schemas
    TimelineEntry, TimelineRegionsResponse, TimelineMunicipalitiesResponse,
    ComparisonItem, ComparisonRegionsResponse,
    ComparisonMatrixCell, ComparisonMatrixResponse,
    HistoricalDataPoint, HistoricalDataResponse,
    # Evidence schemas
    EvidenceStreamItem, EvidenceStreamResponse,
    DocumentReference, DocumentReferencesResponse,
    DocumentSource, DocumentSourcesResponse,
    Citation, CitationsResponse,
    # Export schemas
    ExportFormat, ExportResponse, ReportType, ReportResponse
)


class DeterministicDataGenerator:
    """
    SIN_CARRETA: Generates deterministic sample data for AtroZ dashboard API
    
    All data generation is deterministic based on base seed + entity ID.
    Same seed always produces same data.
    """
    
    def __init__(self, base_seed: int = 42):
        """
        Initialize generator with base seed
        
        Args:
            base_seed: Base seed for all random generation
        """
        self.base_seed = base_seed
        
        # Dimension names
        self.dimension_names = {
            DimensionEnum.D1: "Gobernanza y Participación",
            DimensionEnum.D2: "Desarrollo Económico",
            DimensionEnum.D3: "Infraestructura y Servicios",
            DimensionEnum.D4: "Educación y Cultura",
            DimensionEnum.D5: "Salud y Bienestar",
            DimensionEnum.D6: "Medio Ambiente y Sostenibilidad"
        }
        
        # Policy area names
        self.policy_names = {
            PolicyAreaEnum.P1: "Ordenamiento Social del Territorio",
            PolicyAreaEnum.P2: "Reactivación Económica y Producción Agropecuaria",
            PolicyAreaEnum.P3: "Salud Rural",
            PolicyAreaEnum.P4: "Educación Rural",
            PolicyAreaEnum.P5: "Vivienda, Agua Potable y Saneamiento Básico",
            PolicyAreaEnum.P6: "Infraestructura y Adecuación de Tierras",
            PolicyAreaEnum.P7: "Reincorporación y Convivencia",
            PolicyAreaEnum.P8: "Sistema para la Garantía Progresiva de Derechos",
            PolicyAreaEnum.P9: "Reconciliación y Construcción de Paz",
            PolicyAreaEnum.P10: "Víctimas del Conflicto Armado"
        }
    
    def _get_seed_for_entity(self, entity_id: str) -> int:
        """
        Generate deterministic seed for entity
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Seed integer
        """
        # Hash entity_id to get consistent seed
        hash_val = 0
        for char in entity_id:
            hash_val = (hash_val * 31 + ord(char)) & 0xFFFFFFFF
        return (self.base_seed + hash_val) & 0xFFFFFFFF
    
    def generate_regions(self, count: int = 10) -> List[RegionSummary]:
        """
        Generate list of PDET regions
        
        Args:
            count: Number of regions to generate
            
        Returns:
            List of RegionSummary objects
        """
        rng = create_seeded_generator(self.base_seed)
        
        regions = []
        for i in range(count):
            region_id = f"REGION_{i+1:03d}"
            region_rng = create_seeded_generator(self._get_seed_for_entity(region_id))
            
            lat, lon = region_rng.generate_coordinates()
            
            regions.append(RegionSummary(
                id=region_id,
                name=region_rng.generate_name("Región"),
                coordinates=RegionCoordinates(latitude=lat, longitude=lon),
                overall_score=region_rng.generate_score(30.0, 95.0)
            ))
        
        return regions
    
    def generate_region_detail(self, region_id: str) -> RegionDetail:
        """
        Generate detailed region data
        
        Args:
            region_id: Region ID
            
        Returns:
            RegionDetail object
        """
        rng = create_seeded_generator(self._get_seed_for_entity(region_id))
        
        # Generate dimension scores
        dimension_scores = {}
        for dim in DimensionEnum:
            dimension_scores[dim] = rng.generate_score(25.0, 95.0)
        
        # Generate policy area scores
        policy_scores = {}
        for policy in PolicyAreaEnum:
            policy_scores[policy] = rng.generate_score(20.0, 90.0)
        
        # Calculate overall score as weighted average
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        lat, lon = rng.generate_coordinates()
        
        # Generate metadata
        population = rng.rng.next_int(50000, 500000)
        area_km2 = round(rng.rng.next_float() * 5000 + 1000, 2)
        municipalities_count = rng.rng.next_int(5, 20)
        
        creation_date = (datetime.now() - timedelta(days=rng.rng.next_int(365, 1825))).isoformat()
        
        return RegionDetail(
            id=region_id,
            name=rng.generate_name("Región"),
            coordinates=RegionCoordinates(latitude=lat, longitude=lon),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            policy_area_scores=policy_scores,
            metadata=RegionMetadata(
                population=population,
                area_km2=area_km2,
                municipalities_count=municipalities_count,
                creation_date=creation_date
            ),
            last_updated=datetime.now()
        )
    
    def generate_municipalities(self, region_id: str, count: int = 10) -> List[MunicipalitySummary]:
        """
        Generate list of municipalities for a region
        
        Args:
            region_id: Parent region ID
            count: Number of municipalities
            
        Returns:
            List of MunicipalitySummary objects
        """
        region_rng = create_seeded_generator(self._get_seed_for_entity(region_id))
        
        municipalities = []
        region_num = int(region_id.split("_")[1])
        
        for i in range(count):
            mun_id = f"MUN_{region_num * 100 + i:05d}"
            mun_rng = create_seeded_generator(self._get_seed_for_entity(mun_id))
            
            lat, lon = mun_rng.generate_coordinates()
            
            municipalities.append(MunicipalitySummary(
                id=mun_id,
                name=mun_rng.generate_name("Municipio"),
                region_id=region_id,
                coordinates=RegionCoordinates(latitude=lat, longitude=lon),
                overall_score=mun_rng.generate_score(25.0, 95.0)
            ))
        
        return municipalities
    
    def generate_municipality_detail(self, municipality_id: str) -> MunicipalityDetail:
        """
        Generate detailed municipality data
        
        Args:
            municipality_id: Municipality ID
            
        Returns:
            MunicipalityDetail object
        """
        rng = create_seeded_generator(self._get_seed_for_entity(municipality_id))
        
        # Extract region ID
        mun_num = int(municipality_id.split("_")[1])
        region_num = mun_num // 100
        region_id = f"REGION_{region_num:03d}"
        
        # Generate dimension scores
        dimension_scores = {}
        for dim in DimensionEnum:
            dimension_scores[dim] = rng.generate_score(20.0, 95.0)
        
        # Generate policy area scores
        policy_scores = {}
        for policy in PolicyAreaEnum:
            policy_scores[policy] = rng.generate_score(15.0, 90.0)
        
        # Calculate overall score
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        lat, lon = rng.generate_coordinates()
        
        # Generate metadata
        population = rng.rng.next_int(5000, 100000)
        area_km2 = round(rng.rng.next_float() * 1000 + 100, 2)
        altitude_m = rng.rng.next_int(0, 3000)
        
        return MunicipalityDetail(
            id=municipality_id,
            name=rng.generate_name("Municipio"),
            region_id=region_id,
            coordinates=RegionCoordinates(latitude=lat, longitude=lon),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            policy_area_scores=policy_scores,
            metadata=MunicipalityMetadata(
                population=population,
                area_km2=area_km2,
                altitude_m=altitude_m
            ),
            last_updated=datetime.now()
        )
    
    def generate_question_analysis(
        self,
        policy: PolicyAreaEnum,
        dimension: DimensionEnum,
        question_num: int,
        seed: int
    ) -> QuestionAnalysis:
        """
        Generate analysis for a single question
        
        Args:
            policy: Policy area
            dimension: Dimension
            question_num: Question number (1-5)
            seed: Seed for generation
            
        Returns:
            QuestionAnalysis object
        """
        rng = create_seeded_generator(seed)
        
        question_id = f"{policy.value}-{dimension.value}-Q{question_num}"
        
        # Generate question text
        policy_name = self.policy_names[policy]
        dim_name = self.dimension_names[dimension]
        question_text = f"¿En qué medida el PDM contempla {policy_name} en la dimensión de {dim_name}?"
        
        # Generate score and qualitative level
        score = rng.generate_score(0.0, 3.0)
        
        if score >= 2.5:
            qual_level = QualitativeLevelEnum.EXCELENTE
        elif score >= 2.0:
            qual_level = QualitativeLevelEnum.BUENO
        elif score >= 1.5:
            qual_level = QualitativeLevelEnum.SATISFACTORIO
        elif score >= 1.0:
            qual_level = QualitativeLevelEnum.ACEPTABLE
        elif score >= 0.5:
            qual_level = QualitativeLevelEnum.INSUFICIENTE
        else:
            qual_level = QualitativeLevelEnum.DEFICIENTE
        
        # Generate evidence
        evidence_count = rng.rng.next_int(2, 5)
        evidence_list = []
        
        evidence_templates = [
            "El documento menciona explícitamente la necesidad de fortalecer",
            "Se identifican estrategias concretas para implementar",
            "El plan contempla recursos específicos destinados a",
            "Se establecen metas cuantificables relacionadas con",
            "El PDM incluye mecanismos de seguimiento para"
        ]
        
        for i in range(evidence_count):
            template = rng.choice(evidence_templates)
            evidence_list.append(Evidence(
                text=f"{template} aspectos de {dim_name} en el contexto de {policy_name}.",
                source=f"PDM_{question_id}_Sección_{i+1}",
                confidence=rng.generate_score(0.6, 0.95),
                position=f"pág. {rng.rng.next_int(10, 150)}"
            ))
        
        # Generate explanation
        explanation = f"El análisis de {question_id} revela que el PDM presenta un nivel {qual_level.value.lower()} " \
                     f"de desarrollo en {dim_name} para {policy_name}. La evidencia recopilada sugiere que " \
                     f"{'existen elementos sólidos' if score >= 2.0 else 'se requieren mejoras significativas'} " \
                     f"en la formulación de estrategias y mecanismos de implementación."
        
        return QuestionAnalysis(
            question_id=question_id,
            question_text=question_text,
            qualitative_level=qual_level,
            quantitative_score=score,
            explanation=explanation,
            evidence=evidence_list,
            confidence=rng.generate_score(0.7, 0.95)
        )
    
    def generate_dimension_analysis(
        self,
        dimension: DimensionEnum,
        municipality_id: str
    ) -> DimensionAnalysis:
        """
        Generate analysis for a dimension (5 questions across all policies)
        
        Args:
            dimension: Dimension
            municipality_id: Municipality ID
            
        Returns:
            DimensionAnalysis object
        """
        seed = self._get_seed_for_entity(f"{municipality_id}_{dimension.value}")
        rng = create_seeded_generator(seed)
        
        # Generate 5 questions (one per policy area, cycling through)
        questions = []
        policies = list(PolicyAreaEnum)
        
        for q_num in range(1, 6):
            policy_idx = (q_num - 1) % len(policies)
            policy = policies[policy_idx]
            
            q_seed = self._get_seed_for_entity(f"{municipality_id}_{policy.value}_{dimension.value}_Q{q_num}")
            questions.append(self.generate_question_analysis(policy, dimension, q_num, q_seed))
        
        # Calculate dimension score
        avg_score = sum(q.quantitative_score for q in questions) / len(questions)
        dim_score = (avg_score / 3.0) * 100.0  # Convert to 0-100 scale
        
        # Generate strengths and weaknesses
        strengths = []
        weaknesses = []
        
        strength_templates = [
            f"Cobertura adecuada en {self.dimension_names[dimension]}",
            f"Enfoque integrado en aspectos de {self.dimension_names[dimension]}",
            f"Recursos bien asignados para {self.dimension_names[dimension]}"
        ]
        
        weakness_templates = [
            f"Limitaciones en la articulación de {self.dimension_names[dimension]}",
            f"Falta de indicadores específicos en {self.dimension_names[dimension]}",
            f"Necesidad de fortalecer mecanismos de {self.dimension_names[dimension]}"
        ]
        
        strength_count = rng.rng.next_int(2, 3)
        weakness_count = rng.rng.next_int(2, 3)
        
        for _ in range(strength_count):
            strengths.append(rng.choice(strength_templates))
        
        for _ in range(weakness_count):
            weaknesses.append(rng.choice(weakness_templates))
        
        return DimensionAnalysis(
            dimension_id=dimension,
            dimension_name=self.dimension_names[dimension],
            score=round(dim_score, 2),
            questions=questions,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def generate_clusters(self, region_id: str) -> List[Cluster]:
        """
        Generate cluster analysis for region
        
        Args:
            region_id: Region ID
            
        Returns:
            List of Cluster objects
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"{region_id}_clusters"))
        
        # Generate 3-5 clusters
        cluster_count = rng.rng.next_int(3, 5)
        clusters = []
        
        # Get municipalities for region
        municipalities = self.generate_municipalities(region_id, 10)
        
        cluster_names = [
            "Alto Desempeño Integral",
            "Desarrollo Equilibrado",
            "Necesidades Prioritarias",
            "Recuperación en Progreso",
            "Desafíos Estructurales"
        ]
        
        # Distribute municipalities across clusters
        mun_per_cluster = len(municipalities) // cluster_count
        
        for i in range(cluster_count):
            cluster_id = f"CLUSTER_{i+1:02d}"
            cluster_rng = create_seeded_generator(self._get_seed_for_entity(cluster_id))
            
            # Generate centroid scores
            centroid = {}
            for dim in DimensionEnum:
                centroid[dim] = cluster_rng.generate_score(30.0, 90.0)
            
            # Assign municipalities to cluster
            start_idx = i * mun_per_cluster
            end_idx = start_idx + mun_per_cluster if i < cluster_count - 1 else len(municipalities)
            cluster_muns = municipalities[start_idx:end_idx]
            
            members = []
            for mun in cluster_muns:
                members.append(ClusterMember(
                    municipality_id=mun.id,
                    municipality_name=mun.name,
                    similarity_score=cluster_rng.generate_score(0.7, 0.95)
                ))
            
            # Generate characteristics
            characteristics = [
                f"Centroide en {cluster_names[i % len(cluster_names)]}",
                f"Patrón dominante en {cluster_rng.choice(list(self.dimension_names.values()))}",
                f"{len(members)} municipios con perfil similar"
            ]
            
            clusters.append(Cluster(
                cluster_id=cluster_id,
                cluster_name=cluster_names[i % len(cluster_names)],
                centroid_scores=centroid,
                members=members,
                characteristics=characteristics
            ))
        
        return clusters
    
    # ========================================================================
    # VISUALIZATION GENERATION METHODS
    # ========================================================================
    
    def generate_constellation(self) -> ConstellationVisualizationResponse:
        """
        SIN_CARRETA: Generate constellation visualization
        
        Returns:
            ConstellationVisualizationResponse with nodes and edges
        """
        rng = create_seeded_generator(self._get_seed_for_entity("constellation"))
        
        # Generate nodes for all regions
        nodes = []
        regions = self.generate_regions(10)
        
        for region in regions:
            seed = self._get_seed_for_entity(f"constellation_{region.id}")
            node_rng = create_seeded_generator(seed)
            
            nodes.append(ConstellationNode(
                region_id=region.id,
                region_name=region.name,
                x=node_rng.generate_score(10.0, 90.0),
                y=node_rng.generate_score(10.0, 90.0),
                score=region.overall_score,
                size=node_rng.generate_score(2.0, 8.0)
            ))
        
        # Generate edges (connections between similar regions)
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Only connect some pairs based on deterministic threshold
                edge_seed = self._get_seed_for_entity(f"edge_{nodes[i].region_id}_{nodes[j].region_id}")
                edge_rng = create_seeded_generator(edge_seed)
                
                if edge_rng.rng.next_float() < 0.3:  # 30% connection probability
                    edges.append(ConstellationEdge(
                        source=nodes[i].region_id,
                        target=nodes[j].region_id,
                        strength=edge_rng.generate_score(0.3, 0.9)
                    ))
        
        return ConstellationVisualizationResponse(
            nodes=nodes,
            edges=edges
        )
    
    def generate_phylogram(self, region_id: str) -> PhylogramVisualizationResponse:
        """
        SIN_CARRETA: Generate phylogram (tree) visualization for region
        
        Args:
            region_id: Region ID
            
        Returns:
            PhylogramVisualizationResponse with tree nodes
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"phylogram_{region_id}"))
        
        nodes = []
        
        # Root node (region)
        region_detail = self.generate_region_detail(region_id)
        nodes.append(PhylogramNode(
            id=region_id,
            name=region_detail.name,
            parent_id=None,
            depth=0,
            score=region_detail.overall_score
        ))
        
        # Children nodes (municipalities)
        municipalities = self.generate_municipalities(region_id, 10)
        for mun in municipalities:
            nodes.append(PhylogramNode(
                id=mun.id,
                name=mun.name,
                parent_id=region_id,
                depth=1,
                score=mun.overall_score
            ))
        
        return PhylogramVisualizationResponse(
            region_id=region_id,
            nodes=nodes
        )
    
    def generate_mesh(self, region_id: str) -> MeshVisualizationResponse:
        """
        SIN_CARRETA: Generate 3D mesh visualization for region
        
        Args:
            region_id: Region ID
            
        Returns:
            MeshVisualizationResponse with 3D nodes
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"mesh_{region_id}"))
        
        nodes = []
        municipalities = self.generate_municipalities(region_id, 10)
        
        for mun in municipalities:
            mun_detail = self.generate_municipality_detail(mun.id)
            mun_rng = create_seeded_generator(self._get_seed_for_entity(f"mesh_{mun.id}"))
            
            nodes.append(MeshNode(
                municipality_id=mun.id,
                municipality_name=mun.name,
                x=mun_rng.generate_score(10.0, 90.0),
                y=mun_rng.generate_score(10.0, 90.0),
                z=mun_rng.generate_score(10.0, 90.0),
                dimension_scores=mun_detail.dimension_scores
            ))
        
        return MeshVisualizationResponse(
            region_id=region_id,
            nodes=nodes
        )
    
    def generate_helix(self, municipality_id: str) -> HelixVisualizationResponse:
        """
        SIN_CARRETA: Generate helix visualization for municipality
        
        Args:
            municipality_id: Municipality ID
            
        Returns:
            HelixVisualizationResponse with helix points
        """
        mun_detail = self.generate_municipality_detail(municipality_id)
        
        points = []
        angle_step = 360.0 / 6  # 6 dimensions evenly distributed
        
        for i, (dimension, score) in enumerate(mun_detail.dimension_scores.items()):
            points.append(HelixPoint(
                dimension=dimension,
                dimension_name=self.dimension_names[dimension],
                angle=i * angle_step,
                height=score
            ))
        
        return HelixVisualizationResponse(
            municipality_id=municipality_id,
            municipality_name=mun_detail.name,
            points=points
        )
    
    def generate_radar(self, municipality_id: str) -> RadarVisualizationResponse:
        """
        SIN_CARRETA: Generate radar chart for municipality
        
        Args:
            municipality_id: Municipality ID
            
        Returns:
            RadarVisualizationResponse with radar axes
        """
        mun_detail = self.generate_municipality_detail(municipality_id)
        
        axes = []
        for policy, score in mun_detail.policy_area_scores.items():
            axes.append(RadarAxis(
                policy_area=policy,
                policy_name=self.policy_names[policy],
                score=score
            ))
        
        return RadarVisualizationResponse(
            municipality_id=municipality_id,
            municipality_name=mun_detail.name,
            axes=axes
        )
    
    # ========================================================================
    # TEMPORAL GENERATION METHODS
    # ========================================================================
    
    def generate_timeline_region(self, region_id: str) -> TimelineRegionsResponse:
        """
        SIN_CARRETA: Generate timeline for region
        
        Args:
            region_id: Region ID
            
        Returns:
            TimelineRegionsResponse with timeline events
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"timeline_{region_id}"))
        
        events = []
        event_count = rng.rng.next_int(5, 15)
        
        event_types = [
            "Implementación de Programa",
            "Evaluación de Impacto",
            "Actualización de Política",
            "Hito de Desarrollo",
            "Inversión Pública"
        ]
        
        # Use fixed base date for determinism
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        start_date = base_date - timedelta(days=rng.rng.next_int(365, 1825))
        
        for i in range(event_count):
            event_date = start_date + timedelta(days=i * rng.rng.next_int(30, 90))
            
            events.append(TimelineEntry(
                timestamp=event_date.isoformat(),
                event_type=rng.choice(event_types),
                description=f"Evento significativo en el desarrollo de {region_id} relacionado con implementación PDET.",
                score=rng.generate_score(40.0, 95.0) if rng.rng.next_float() > 0.3 else None
            ))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        return TimelineRegionsResponse(
            region_id=region_id,
            events=events,
            start_date=events[0].timestamp if events else base_date.isoformat(),
            end_date=events[-1].timestamp if events else base_date.isoformat()
        )
    
    def generate_timeline_municipality(self, municipality_id: str) -> TimelineMunicipalitiesResponse:
        """
        SIN_CARRETA: Generate timeline for municipality
        
        Args:
            municipality_id: Municipality ID
            
        Returns:
            TimelineMunicipalitiesResponse with timeline events
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"timeline_{municipality_id}"))
        
        events = []
        event_count = rng.rng.next_int(8, 20)
        
        event_types = [
            "Proyecto Comunitario",
            "Actualización PDM",
            "Inversión Local",
            "Programa Social",
            "Mejora de Infraestructura"
        ]
        
        # Use fixed base date for determinism
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        start_date = base_date - timedelta(days=rng.rng.next_int(365, 1460))
        
        for i in range(event_count):
            event_date = start_date + timedelta(days=i * rng.rng.next_int(20, 60))
            
            events.append(TimelineEntry(
                timestamp=event_date.isoformat(),
                event_type=rng.choice(event_types),
                description=f"Actividad importante en {municipality_id} dentro del marco PDET.",
                score=rng.generate_score(35.0, 90.0) if rng.rng.next_float() > 0.4 else None
            ))
        
        events.sort(key=lambda e: e.timestamp)
        
        return TimelineMunicipalitiesResponse(
            municipality_id=municipality_id,
            events=events,
            start_date=events[0].timestamp if events else base_date.isoformat(),
            end_date=events[-1].timestamp if events else base_date.isoformat()
        )
    
    def generate_comparison_regions(self) -> ComparisonRegionsResponse:
        """
        SIN_CARRETA: Generate comparison of all regions
        
        Returns:
            ComparisonRegionsResponse with comparison items
        """
        regions = self.generate_regions(10)
        items = []
        
        for region_summary in regions:
            region_detail = self.generate_region_detail(region_summary.id)
            items.append(ComparisonItem(
                entity_id=region_detail.id,
                entity_name=region_detail.name,
                dimension_scores=region_detail.dimension_scores,
                overall_score=region_detail.overall_score
            ))
        
        return ComparisonRegionsResponse(items=items)
    
    def generate_comparison_matrix(
        self,
        entity_ids: List[str],
        dimensions: Optional[List[DimensionEnum]] = None
    ) -> ComparisonMatrixResponse:
        """
        SIN_CARRETA: Generate comparison matrix
        
        Args:
            entity_ids: List of entity IDs to compare
            dimensions: Optional specific dimensions to compare
            
        Returns:
            ComparisonMatrixResponse with similarity matrix
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"matrix_{'_'.join(entity_ids)}"))
        
        matrix = []
        
        for i, entity_i in enumerate(entity_ids):
            for j, entity_j in enumerate(entity_ids):
                if i == j:
                    similarity = 1.0
                else:
                    # Generate deterministic similarity based on entity pair
                    pair_seed = self._get_seed_for_entity(f"similarity_{entity_i}_{entity_j}")
                    pair_rng = create_seeded_generator(pair_seed)
                    similarity = pair_rng.generate_score(0.2, 0.9)
                
                matrix.append(ComparisonMatrixCell(
                    row_entity=entity_i,
                    col_entity=entity_j,
                    similarity=similarity
                ))
        
        return ComparisonMatrixResponse(
            entity_ids=entity_ids,
            matrix=matrix
        )
    
    def generate_historical_data(
        self,
        entity_type: str,
        entity_id: str,
        start_year: int,
        end_year: int
    ) -> HistoricalDataResponse:
        """
        SIN_CARRETA: Generate historical data for entity
        
        Args:
            entity_type: Type of entity (region or municipality)
            entity_id: Entity ID
            start_year: Start year
            end_year: End year
            
        Returns:
            HistoricalDataResponse with historical data points
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"historical_{entity_id}"))
        
        data_points = []
        
        for year in range(start_year, end_year + 1):
            year_rng = create_seeded_generator(self._get_seed_for_entity(f"{entity_id}_{year}"))
            
            # Generate dimension scores with some trend
            dimension_scores = {}
            for dim in DimensionEnum:
                # Add slight upward trend over years
                base_score = year_rng.generate_score(30.0, 85.0)
                trend_adjustment = (year - start_year) * 1.5
                score = min(100.0, base_score + trend_adjustment)
                dimension_scores[dim] = round(score, 2)
            
            overall_score = sum(dimension_scores.values()) / len(dimension_scores)
            
            data_points.append(HistoricalDataPoint(
                year=year,
                dimension_scores=dimension_scores,
                overall_score=round(overall_score, 2)
            ))
        
        return HistoricalDataResponse(
            entity_type=entity_type,
            entity_id=entity_id,
            data_points=data_points,
            start_year=start_year,
            end_year=end_year
        )
    
    # ========================================================================
    # EVIDENCE GENERATION METHODS
    # ========================================================================
    
    def generate_evidence_stream(self, page: int, per_page: int) -> EvidenceStreamResponse:
        """
        SIN_CARRETA: Generate evidence stream
        
        Args:
            page: Page number
            per_page: Items per page
            
        Returns:
            EvidenceStreamResponse with evidence items
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"evidence_stream_p{page}"))
        
        total = 500  # Total evidence items available
        items = []
        
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total)
        
        # Use fixed base date for determinism
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        
        for i in range(start_idx, end_idx):
            item_rng = create_seeded_generator(self._get_seed_for_entity(f"evidence_{i:06d}"))
            
            timestamp = base_date - timedelta(days=item_rng.rng.next_int(0, 365))
            
            items.append(EvidenceStreamItem(
                evidence_id=f"EV_{i:06d}",
                text=f"Evidencia documental relacionada con implementación PDET que demuestra avances en indicadores clave de desarrollo territorial.",
                source=f"Documento_{item_rng.rng.next_int(1, 100):03d}.pdf",
                confidence=item_rng.generate_score(0.6, 0.95),
                timestamp=timestamp.isoformat(),
                entity_id=f"REGION_{item_rng.rng.next_int(1, 10):03d}"
            ))
        
        return EvidenceStreamResponse(
            items=items,
            total=total,
            page=page,
            per_page=per_page
        )
    
    def generate_document_references(self, region_id: str) -> DocumentReferencesResponse:
        """
        SIN_CARRETA: Generate document references for region
        
        Args:
            region_id: Region ID
            
        Returns:
            DocumentReferencesResponse with document references
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"docs_{region_id}"))
        
        references = []
        ref_count = rng.rng.next_int(10, 25)
        
        authors = ["García, J.", "Martínez, A.", "López, M.", "Rodríguez, C.", "Pérez, L."]
        
        # Use fixed base date for determinism
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        
        for i in range(ref_count):
            ref_rng = create_seeded_generator(self._get_seed_for_entity(f"{region_id}_doc_{i}"))
            
            date = base_date - timedelta(days=ref_rng.rng.next_int(0, 1825))
            
            references.append(DocumentReference(
                document_id=f"DOC_{ref_rng.rng.next_int(100000, 999999):06d}",
                title=f"Análisis de Implementación PDET en {region_id}",
                author=ref_rng.choice(authors),
                date=date.isoformat(),
                url=f"https://pdet.gov.co/docs/{region_id.lower()}/doc_{i}.pdf"
            ))
        
        return DocumentReferencesResponse(
            region_id=region_id,
            references=references,
            total=len(references)
        )
    
    def generate_document_sources(self, question_id: str) -> DocumentSourcesResponse:
        """
        SIN_CARRETA: Generate document sources for question
        
        Args:
            question_id: Question ID
            
        Returns:
            DocumentSourcesResponse with document sources
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"sources_{question_id}"))
        
        sources = []
        source_count = rng.rng.next_int(3, 8)
        
        for i in range(source_count):
            src_rng = create_seeded_generator(self._get_seed_for_entity(f"{question_id}_src_{i}"))
            
            sources.append(DocumentSource(
                source_id=f"SRC_{src_rng.rng.next_int(100000, 999999):06d}",
                document_title=f"Plan de Desarrollo Municipal - Sección {i+1}",
                excerpt=f"Extracto relevante del documento que aborda aspectos específicos de la pregunta {question_id}.",
                page_number=src_rng.rng.next_int(10, 200),
                relevance=src_rng.generate_score(0.6, 0.95)
            ))
        
        return DocumentSourcesResponse(
            question_id=question_id,
            sources=sources,
            total=len(sources)
        )
    
    def generate_citations(self, indicator_id: str) -> CitationsResponse:
        """
        SIN_CARRETA: Generate citations for indicator
        
        Args:
            indicator_id: Indicator ID
            
        Returns:
            CitationsResponse with citations
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"citations_{indicator_id}"))
        
        citations = []
        cit_count = rng.rng.next_int(5, 12)
        
        authors = ["García", "Martínez", "López", "Rodríguez", "Pérez"]
        
        for i in range(cit_count):
            cit_rng = create_seeded_generator(self._get_seed_for_entity(f"{indicator_id}_cit_{i}"))
            
            year = cit_rng.rng.next_int(2015, 2023)
            author = cit_rng.choice(authors)
            
            citations.append(Citation(
                citation_id=f"CIT_{cit_rng.rng.next_int(100000, 999999):06d}",
                text=f"Estudio sobre indicadores de desarrollo territorial en regiones PDET.",
                source=f"Revista de Desarrollo Territorial, Vol. {cit_rng.rng.next_int(1, 20)}",
                year=year,
                citation_format=f"{author}, J. ({year}). Análisis de indicadores PDET. Revista de Desarrollo Territorial."
            ))
        
        return CitationsResponse(
            indicator_id=indicator_id,
            citations=citations,
            total=len(citations)
        )
    
    # ========================================================================
    # EXPORT GENERATION METHODS
    # ========================================================================
    
    def generate_export_dashboard(
        self,
        format: ExportFormat,
        include_visualizations: bool,
        include_raw_data: bool
    ) -> ExportResponse:
        """
        SIN_CARRETA: Generate dashboard export
        
        Args:
            format: Export format
            include_visualizations: Include visualizations
            include_raw_data: Include raw data
            
        Returns:
            ExportResponse with export information
        """
        rng = create_seeded_generator(self._get_seed_for_entity("export_dashboard"))
        
        export_id = f"EXP_{rng.rng.next_int(10000000, 99999999):08d}"
        expires_at = datetime.now() + timedelta(hours=24)
        
        # Estimate file size based on options
        base_size = 5 * 1024 * 1024  # 5MB base
        if include_visualizations:
            base_size += 10 * 1024 * 1024
        if include_raw_data:
            base_size += 20 * 1024 * 1024
        
        return ExportResponse(
            export_id=export_id,
            format=format,
            download_url=f"https://api.pdet.gov.co/downloads/{export_id}.{format.value}",
            expires_at=expires_at.isoformat(),
            size_bytes=base_size
        )
    
    def generate_export_region(
        self,
        region_id: str,
        format: ExportFormat,
        include_municipalities: bool,
        include_analysis: bool
    ) -> ExportResponse:
        """
        SIN_CARRETA: Generate region export
        
        Args:
            region_id: Region ID
            format: Export format
            include_municipalities: Include municipalities
            include_analysis: Include analysis
            
        Returns:
            ExportResponse with export information
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"export_{region_id}"))
        
        export_id = f"EXP_{rng.rng.next_int(10000000, 99999999):08d}"
        expires_at = datetime.now() + timedelta(hours=24)
        
        base_size = 2 * 1024 * 1024
        if include_municipalities:
            base_size += 5 * 1024 * 1024
        if include_analysis:
            base_size += 8 * 1024 * 1024
        
        return ExportResponse(
            export_id=export_id,
            format=format,
            download_url=f"https://api.pdet.gov.co/downloads/{export_id}.{format.value}",
            expires_at=expires_at.isoformat(),
            size_bytes=base_size
        )
    
    def generate_export_comparison(
        self,
        entity_ids: List[str],
        format: ExportFormat,
        dimensions: Optional[List[DimensionEnum]]
    ) -> ExportResponse:
        """
        SIN_CARRETA: Generate comparison export
        
        Args:
            entity_ids: List of entity IDs
            format: Export format
            dimensions: Optional specific dimensions
            
        Returns:
            ExportResponse with export information
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"export_comp_{'_'.join(entity_ids)}"))
        
        export_id = f"EXP_{rng.rng.next_int(10000000, 99999999):08d}"
        expires_at = datetime.now() + timedelta(hours=24)
        
        base_size = len(entity_ids) * 1024 * 1024
        
        return ExportResponse(
            export_id=export_id,
            format=format,
            download_url=f"https://api.pdet.gov.co/downloads/{export_id}.{format.value}",
            expires_at=expires_at.isoformat(),
            size_bytes=base_size
        )
    
    def generate_standard_report(self, report_type: ReportType) -> ReportResponse:
        """
        SIN_CARRETA: Generate standard report
        
        Args:
            report_type: Type of report
            
        Returns:
            ReportResponse with report information
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"report_{report_type.value}"))
        
        report_id = f"RPT_{rng.rng.next_int(10000000, 99999999):08d}"
        expires_at = datetime.now() + timedelta(hours=48)
        
        size_map = {
            ReportType.EXECUTIVE_SUMMARY: 3 * 1024 * 1024,
            ReportType.DETAILED_ANALYSIS: 15 * 1024 * 1024,
            ReportType.COMPARISON: 8 * 1024 * 1024,
            ReportType.TRENDS: 10 * 1024 * 1024
        }
        
        return ReportResponse(
            report_id=report_id,
            report_type=report_type.value,
            download_url=f"https://api.pdet.gov.co/reports/{report_id}.pdf",
            expires_at=expires_at.isoformat(),
            size_bytes=size_map.get(report_type, 5 * 1024 * 1024)
        )
    
    def generate_custom_report(
        self,
        title: str,
        entity_ids: List[str],
        sections: List[str],
        format: ExportFormat
    ) -> ReportResponse:
        """
        SIN_CARRETA: Generate custom report
        
        Args:
            title: Report title
            entity_ids: Entity IDs to include
            sections: Report sections
            format: Report format
            
        Returns:
            ReportResponse with report information
        """
        rng = create_seeded_generator(self._get_seed_for_entity(f"custom_{title}"))
        
        report_id = f"RPT_{rng.rng.next_int(10000000, 99999999):08d}"
        expires_at = datetime.now() + timedelta(hours=48)
        
        base_size = (len(entity_ids) * len(sections) * 2) * 1024 * 1024
        
        return ReportResponse(
            report_id=report_id,
            report_type="custom",
            download_url=f"https://api.pdet.gov.co/reports/{report_id}.{format.value}",
            expires_at=expires_at.isoformat(),
            size_bytes=base_size
        )


# Singleton instance
_generator_instance = None


def get_data_generator(base_seed: int = 42) -> DeterministicDataGenerator:
    """
    SIN_CARRETA: Get or create singleton data generator instance
    
    Args:
        base_seed: Base seed (only used on first call)
        
    Returns:
        DeterministicDataGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DeterministicDataGenerator(base_seed)
    return _generator_instance
