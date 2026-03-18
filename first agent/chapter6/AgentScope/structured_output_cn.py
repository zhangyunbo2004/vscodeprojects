# -*- coding: utf-8 -*-
"""三国狼人杀游戏的结构化输出模型"""
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, validator
from agentscope.agent import AgentBase


class DiscussionModelCN(BaseModel):
    """中文版讨论输出格式"""
    
    reach_agreement: bool = Field(
        description="是否已达成一致意见",
    )
    confidence_level: int = Field(
        description="对当前推理的信心程度(1-10)",
        ge=1, le=10
    )
    key_evidence: Optional[str] = Field(
        description="支持你观点的关键证据",
        default=None
    )


def get_vote_model_cn(agents: list[AgentBase]) -> type[BaseModel]:
    """获取中文版投票模型"""
    
    # 获取玩家名称列表用于验证
    player_names = [_.name for _ in agents]
    
    class VoteModelCN(BaseModel):
        """中文版投票输出格式"""
        
        # 将 Literal 改为 str
        vote: str = Field(
            description="你要投票淘汰的玩家姓名",
        )
        reason: str = Field(
            description="投票理由，简要说明为什么选择此人",
        )
        suspicion_level: int = Field(
            description="对被投票者的怀疑程度(1-10)",
            ge=1, le=10
        )
        
        # 添加验证器
        @validator('vote')
        def validate_vote(cls, v):
            if v not in player_names:
                raise ValueError(f'无效的投票目标: {v}，有效玩家: {", ".join(player_names)}')
            return v
    
    return VoteModelCN


class WitchActionModelCN(BaseModel):
    """中文版女巫行动模型"""
    
    use_antidote: bool = Field(
        description="是否使用解药救人",
        default=False
    )
    use_poison: bool = Field(
        description="是否使用毒药杀人", 
        default=False
    )
    target_name: Optional[str] = Field(
        description="目标玩家姓名（救人或毒杀的对象）",
        default=None
    )
    action_reason: Optional[str] = Field(
        description="行动理由",
        default=None
    )
    
    @validator('target_name')
    def validate_target_name(cls, v, values):
        """验证目标名称"""
        if (values.get('use_antidote') or values.get('use_poison')) and not v:
            raise ValueError('使用解药或毒药时必须指定目标玩家')
        return v


def get_seer_model_cn(agents: list[AgentBase]) -> type[BaseModel]:
    """获取中文版预言家模型"""
    
    # 获取玩家名称列表用于验证
    player_names = [_.name for _ in agents]
    
    class SeerModelCN(BaseModel):
        """中文版预言家查验格式"""
        
        # 将 Literal 改为 str
        target: str = Field(
            description="要查验的玩家姓名",
        )
        check_reason: str = Field(
            description="查验此人的原因",
        )
        priority_level: int = Field(
            description="查验优先级(1-10)",
            ge=1, le=10
        )
        
        # 添加验证器
        @validator('target')
        def validate_target(cls, v):
            if v not in player_names:
                raise ValueError(f'无效的查验目标: {v}，有效玩家: {", ".join(player_names)}')
            return v
    
    return SeerModelCN


def get_hunter_model_cn(agents: list[AgentBase]) -> type[BaseModel]:
    """获取中文版猎人模型"""
    
    # 获取玩家名称列表用于验证
    player_names = [_.name for _ in agents]
    
    class HunterModelCN(BaseModel):
        """中文版猎人开枪格式"""
        
        shoot: bool = Field(
            description="是否使用开枪技能",
        )
        # 将 Optional[Literal] 改为 Optional[str]
        target: Optional[str] = Field(
            description="开枪目标玩家姓名",
            default=None
        )
        shoot_reason: Optional[str] = Field(
            description="开枪理由",
            default=None
        )
        
        # 添加验证器
        @validator('target')
        def validate_target(cls, v, values):
            if values.get('shoot') and not v:
                raise ValueError('开枪时必须指定目标')
            if v and v not in player_names:
                raise ValueError(f'无效的开枪目标: {v}，有效玩家: {", ".join(player_names)}')
            return v
    
    return HunterModelCN


class WerewolfKillModelCN(BaseModel):
    """中文版狼人击杀模型"""
    
    target: str = Field(
        description="要击杀的玩家姓名",
    )
    kill_strategy: str = Field(
        description="击杀策略说明",
    )
    team_coordination: Optional[str] = Field(
        description="与狼队友的配合计划",
        default=None
    )


class GameAnalysisModelCN(BaseModel):
    """中文版游戏分析模型"""
    
    suspected_werewolves: List[str] = Field(
        description="怀疑的狼人名单",
        default_factory=list
    )
    trusted_players: List[str] = Field(
        description="信任的玩家名单", 
        default_factory=list
    )
    key_clues: List[str] = Field(
        description="关键线索列表",
        default_factory=list
    )
    next_strategy: str = Field(
        description="下一步策略",
    )