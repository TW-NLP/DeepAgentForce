"""
用户偏好挖掘系统
1. LLM 提取对话中的实体和关系
2. NetworkX 构建知识图谱
3. 图算法（PageRank、中心性等）挖掘用户偏好
"""

import json
import os
from langchain_openai import ChatOpenAI
import networkx as nx
from typing import List, Dict, Tuple



class UserPreferenceMining():
    """用户偏好挖掘,定时任务"""
    
    def __init__(self,settings):
        self.settings = settings
        self.client = ChatOpenAI(base_url=settings.LLM_URL, api_key=settings.LLM_API_KEY, model=settings.LLM_MODEL)

        self.graph = nx.DiGraph()
        
        # 添加用户中心节点
        self.graph.add_node('USER', type='user', label='用户')

        
    
    def extract_entities_relations(self, conversations: List[Dict]) -> List[Dict]:
        """
        步骤1: 使用 LLM 从对话中提取实体和关系
        
        Args:
            conversations: 对话列表
            
        Returns:
            提取的实体关系列表
        """
        # 构建对话文本
        conv_text = "\n".join(conversations)
        
        prompt = f"""从用户的对话中提取实体和关系，用于构建知识图谱。

对话记录：
{conv_text}

请提取：
1. 实体（entities）：用户提到的具体事物、概念、话题等
2. 关系（relations）：用户与实体之间的关系

以 JSON 格式返回：
{{
  "entities": [
    {{"name": "Python", "type": "技术", "mentions": 3}},
    {{"name": "数据分析", "type": "领域", "mentions": 2}}
  ],
  "relations": [
    {{"source": "USER", "target": "Python", "relation": "感兴趣", "weight": 0.9}},
    {{"source": "USER", "target": "数据分析", "relation": "想学习", "weight": 0.8}}
  ]
}}

注意：
- 实体 name 要简洁明确
- 关系 source 统一用 "USER" 代表用户
- weight 范围 0-1，表示关系强度
- 只提取明确的、有价值的实体和关系

只返回 JSON，无其他内容。"""

        print("🤖 LLM 提取实体和关系...")
        response = self.client.invoke(prompt)
        
        result_text = response.content
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        
        extracted = json.loads(result_text)
        print(f"✓ 提取到 {len(extracted['entities'])} 个实体")
        print(f"✓ 提取到 {len(extracted['relations'])} 个关系")
        
        return extracted
    
    def build_knowledge_graph(self, extracted_data: Dict):
        """
        步骤2: 使用 NetworkX 构建知识图谱
        
        Args:
            extracted_data: LLM 提取的实体和关系
        """
        print("\n📊 构建知识图谱...")
        
        # 添加实体节点
        for entity in extracted_data['entities']:
            self.graph.add_node(
                entity['name'],
                type='entity',
                entity_type=entity.get('type', 'unknown'),
                mentions=entity.get('mentions', 1),
                label=entity['name']
            )
        
        # 添加关系边
        for relation in extracted_data['relations']:
            self.graph.add_edge(
                relation['source'],
                relation['target'],
                relation=relation['relation'],
                weight=relation.get('weight', 0.5)
            )
        
        print(f"✓ 图谱节点数: {self.graph.number_of_nodes()}")
        print(f"✓ 图谱边数: {self.graph.number_of_edges()}")
    
    def mine_preferences_with_graph_algorithms(self) -> Dict:
        """
        步骤3: 使用图算法挖掘用户偏好
        
        使用的算法：
        - PageRank: 计算节点重要性
        - Degree Centrality: 度中心性（连接数）
        - Betweenness Centrality: 介数中心性
        """
        print("\n🔍 使用图算法挖掘偏好...")

        
        # 1. PageRank - 计算每个实体的重要性
        pagerank_scores = nx.pagerank(self.graph, weight='weight')
        # 排除用户节点，只看实体
        entity_pagerank = {
            node: score 
            for node, score in pagerank_scores.items() 
            if node != 'USER'
        }
        top_pagerank = sorted(entity_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 2. 度中心性 - 用户直接连接的实体
        user_neighbors = list(self.graph.successors('USER'))
        neighbor_weights = {}
        for neighbor in user_neighbors:
            edge_data = self.graph['USER'][neighbor]
            neighbor_weights[neighbor] = edge_data.get('weight', 0.5)
        
        top_neighbors = sorted(neighbor_weights.items(), key=lambda x: x[1], reverse=True)
        
        # 3. 节点属性分析
        entity_mentions = {}
        entity_types = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'entity':
                entity_mentions[node] = data.get('mentions', 0)
                entity_types[node] = data.get('entity_type', 'unknown')
        
        top_mentions = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 4. 综合计算偏好得分
        preference_scores = {}
        for entity in entity_pagerank.keys():
            score = 0.0
            
            # PageRank 权重 (40%)
            score += pagerank_scores.get(entity, 0) * 40
            
            # 直接连接权重 (30%)
            if entity in neighbor_weights:
                score += neighbor_weights[entity] * 30
            
            # 提及次数权重 (30%)
            mentions = entity_mentions.get(entity, 0)
            max_mentions = max(entity_mentions.values()) if entity_mentions else 1
            score += (mentions / max_mentions) * 30
            
            preference_scores[entity] = score
        
        top_preferences = sorted(preference_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 整理结果
        result = {
            "top_preferences": [
                {
                    "entity": entity,
                    "score": round(score, 3),
                    "type": entity_types.get(entity, 'unknown'),
                    "mentions": entity_mentions.get(entity, 0),
                    "pagerank": round(pagerank_scores.get(entity, 0), 4)
                }
                for entity, score in top_preferences
            ],
            "algorithm_results": {
                "pagerank_top10": [
                    {"entity": e, "score": round(s, 4)} 
                    for e, s in top_pagerank
                ],
                "direct_connections": [
                    {"entity": e, "weight": round(w, 3)}
                    for e, w in top_neighbors
                ],
                "most_mentioned": [
                    {"entity": e, "mentions": m}
                    for e, m in top_mentions
                ]
            },
            "graph_statistics": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "user_connections": self.graph.out_degree('USER'),
                "avg_clustering": round(nx.average_clustering(self.graph.to_undirected()), 3)
            }
        }
        
        return result
    
    def save_graph(self, filepath: str = 'knowledge_graph.json'):
        """保存知识图谱"""
        graph_data = {
            'nodes': [
                {'id': node, **data}
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 知识图谱已保存: {filepath}")

    def get_frontend_format(self) -> Dict:
        return json.loads(open(self.settings.PERSON_LIKE_FILE, 'r', encoding='utf-8').read())
    def person_like_save(self) -> Dict:
        """
        返回符合前端要求的数据格式
        用于 /get/person_like 接口
        """
        # 获取session 数据
        sessions_list=[]
        for file_i in os.listdir(self.settings.HISTORY_DIR):
            file_path = os.path.join(self.settings.HISTORY_DIR, file_i)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data=json.load(f)
                sessions_list.extend([i.get("user_content", "") for i in file_data.get('conversations', [])])
        sessions_list = [item for item in sessions_list if item.strip()]

        # 步骤1: LLM 提取实体和关系
        extracted = self.extract_entities_relations(sessions_list)
        # 步骤2: 构建知识图谱
        self.build_knowledge_graph(extracted)
        # 挖掘偏好
        preferences_result = self.mine_preferences_with_graph_algorithms()
        
        # 图谱数据
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    'type': data.get('type', 'unknown'),
                    'label': data.get('label', node)
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'relation': data.get('relation', ''),
                    'weight': data.get('weight', 0.5)
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        api_response = {
            'graph': graph_data,
            'preferences': preferences_result['top_preferences'],
            'statistics': preferences_result['graph_statistics'],
        }

        prompt = f"""
请根据以下“用户偏好挖掘系统”的输出数据，生成一段简练、专业的【用户画像侧写】。

【输入数据】：
{json.dumps(api_response, ensure_ascii=False, indent=2)}

【要求】：
1. **核心定位**：一句话概括用户的核心身份（例如：AI方向的开发者、Python初学者等）。
2. **偏好解读**：结合 `preferences` 中的分数排名，说明用户最关注的领域或工具。
3. **关系细节**：利用 `edges` 中的 `relation` 字段（如“感兴趣”vs“偏好使用”）来区分用户是单纯感兴趣还是有实操需求。
4. **口吻**：客观、专业，类似于CRM系统中的用户备注。
5. **字数**：150字以内。
"""
        api_response['summary']=self.client.invoke(prompt).content.strip()
        with open(self.settings.PERSON_LIKE_FILE, 'w', encoding='utf-8') as f:
            json.dump(api_response, f, ensure_ascii=False, indent=2)
    
    def visualize_graph(self, output_path: str = 'preference_graph.png'):
        """可视化知识图谱"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 10))
            
            # 布局
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
            
            # 节点分组
            user_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'user']
            entity_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'entity']
            
            # 绘制节点
            nx.draw_networkx_nodes(self.graph, pos, nodelist=user_nodes, 
                                  node_color='#FF6B6B', node_size=3000, alpha=0.9, label='用户')
            nx.draw_networkx_nodes(self.graph, pos, nodelist=entity_nodes,
                                  node_color='#4ECDC4', node_size=1500, alpha=0.8, label='实体')
            
            # 绘制边
            nx.draw_networkx_edges(self.graph, pos, edge_color='gray', 
                                  alpha=0.3, arrows=True, arrowsize=20, width=2)
            
            # 绘制标签
            nx.draw_networkx_labels(self.graph, pos, font_size=9, font_family='sans-serif')
            
            plt.title('用户偏好知识图谱', fontsize=16, fontweight='bold')
            plt.legend(loc='upper left', fontsize=11)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"🎨 图谱可视化已保存: {output_path}")
        except Exception as e:
            print(f"可视化失败: {e}")

