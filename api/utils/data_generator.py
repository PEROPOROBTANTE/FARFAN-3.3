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

from typing import List, Dict
from datetime import datetime, timedelta
from api.utils.seeded_rng import create_seeded_generator
from api.models.schemas import (
    RegionSummary, RegionDetail, RegionCoordinates, RegionMetadata,
    MunicipalitySummary, MunicipalityDetail, MunicipalityMetadata,
    QuestionAnalysis, DimensionAnalysis, Evidence,
    Cluster, ClusterMember,
    QualitativeLevelEnum, DimensionEnum, PolicyAreaEnum
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
