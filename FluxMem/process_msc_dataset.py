'''
DATASET FORMAT:
12 FEATURES (as mentioned in FluxMem paper):
0. page_count 
1. avg_page_length
2. entity_density
3. relation_indicators
4. topic_diversity
5. topic_transitions
6. is_qna_pattern
7. is_decision_tree
8. is_entity centric
9. time_span
10. temporal_density
11. semantic_complexity

+2 extra features:
12. hyperedge_density
13. visual_salience_score

1 CLASS LABEL:
0 is hypergraph
1 is visual canvas
2 is vector store
'''

import numpy as np
import pandas as pd
from memory_structures import Page
from feature_extractor import FeatureExtractor
from best_structure_evaluator import BestStructureEvaluator
from datasets import load_dataset

import logging
logging.basicConfig(
    filename='data_processing.log', 
    filemode='w', 
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from qwen_client import QwenClient, QwenConfig
qwen = QwenClient.load(config=QwenConfig(model_path="Qwen/Qwen3-4B", enable_thinking=False))

class DatasetEvaluator:
    SECONDS_TO_READ_PER_WORD = 0.25
    SECONDS_TO_WRITE_PER_WORD = 1
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.structure_evaluator = BestStructureEvaluator()
        self.all_rows = []
        self.topic_hyperedges = {'dialoug_id': 0, 'content': ""}

    def construct_page_list(self, dialogue: list[str], speakers: list[str]) -> list[Page]:
        ep = []
        n = len(dialogue)
        timestamp=0
        i=0
        while i<n:
            user_text = ""
            if (i<n and speakers[i] == 'Speaker 1'):
                user_text = dialogue[i]
                i += 1
            agent_text = ""
            if (i<n and speakers[i] == 'Speaker 2'):
                agent_text = dialogue[i]
                i += 1
            timestamp += self.SECONDS_TO_WRITE_PER_WORD*len(user_text.split()) + self.SECONDS_TO_READ_PER_WORD*len(agent_text.split())
            ep.append(Page(user_text=user_text, agent_text=agent_text, timestamp=timestamp ,embedding=qwen.embed("USER: " + user_text + "\nAGENT: " + agent_text)))
        return ep

    def get_features_dict(self, ep: list[Page], info: dict) -> dict:
        features = self.feature_extractor.extract_features(ep, info)
        feature_dict = {
            'page_count': features[0],
            'avg_page_length': features[1],
            'entity_density': features[2],
            'relation_indicators': features[3],
            'topic_diversity': features[4],
            'topic_transitions': features[5],
            'is_qna_pattern': features[6],
            'is_decision_tree': features[7],
            'is_entity centric': features[8],
            'time_span': features[9],
            'temporal_density': features[10],
            'semantic_complexity': features[11],
            'hyperedge_density': features[12],
            'visual_salience_score': features[13]
            }
        return feature_dict
    
    def get_label_dict(self, ep: list[Page], persona1: list[str], persona2: list[str]) -> dict:
        label = self.structure_evaluator.find_best_structure(ep, persona1, persona2)
        label_dict = {
            'structure': label
        }
        return label_dict

    def eval_row(self, row):
        if (self.topic_hyperedges["dialoug_id"] != row["dialoug_id"]):
                    self.topic_hyperedges["dialoug_id"] = row["dialoug_id"]
                    self.topic_hyperedges["content"] = "" 
        ep = self.construct_page_list(dialogue=row["dialogue"], speakers=row["speaker"])
        try:
            info = self.feature_extractor.get_info_json(ep)
            feature_dict = self.get_features_dict(ep, info)
            label_dict = self.get_label_dict(ep, row["persona1"], row["persona2"])
        except Exception as e:
            logger.error(f"Dialogue {row["dialoug_id"]}, Session: {row["session_id"]} data extraction failed. MSG: {str(e)}")
            return
        res = feature_dict | label_dict
        self.all_rows.append(res)

    def multi_rows_feature_ex(self, rows):
        import json
        all_page_lists = []
        all_prompts = []
        for row in rows.itertuples():
          ep = self.construct_page_list(dialogue=row.dialogue, speakers=row.speaker)
          all_page_lists.append(ep)
          all_prompts.append(self.feature_extractor.build_user_prompt(ep))
        responses = qwen.generate_batch(system=FeatureExtractor.system_prompt, prompts=all_prompts, max_new_tokens=2048)
        for i in range(len(rows)):
          info = None
          try:
            info = json.loads(responses[i].text)
          except Exception as e:
            logger.warning(f"RETRYING #{i} Dialogue {rows.iloc[i, 1]}, Session: {rows.iloc[i, 2]}. MSG: {str(e)}\n LLM Response: {responses[i].text}\n")
            try:
              retry_res = qwen.chat(user=all_prompts[i], system=FeatureExtractor.system_prompt_reduced_info, max_tokens=2048)
              info = json.loads(retry_res)
            except Exception as e:
              logger.error(f"#{i} Dialogue {rows.iloc[i, 1]}, Session: {rows.iloc[i, 2]} data extraction failed. MSG: {str(e)}\n LLM Response: {retry_res}\n")
              continue
          self.all_rows.append(self.get_features_dict(all_page_lists[i], info))

    def eval_label_only(row):
        ep = self.construct_page_list(dialogue=row["dialogue"], speakers=row["speaker"])
        try:
          label_dict = self.get_label_dict(ep, row["persona1"], row["persona2"])
        except Exception as e:
            logger.error(f"Dialogue {row["dialoug_id"]}, Session: {row["session_id"]} data extraction failed. MSG: {str(e)}")
            return
        self.all_rows.append(label_dict)

    def save_to_csv(self, fileName: str):
        pd.DataFrame(self.all_rows).to_csv(fileName, index=False)


def main():
    dataset = load_dataset("nayohan/multi_session_chat")

    for w in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[w])
        df['dialoug_id'] = pd.to_numeric(df['dialoug_id'])
        df['session_id'] = pd.to_numeric(df['session_id'])
        df = df.sort_values(by=["dialoug_id", "session_id"]).reset_index(drop=True)
        dataset_evaluator = DatasetEvaluator()
        #df.iloc[0:5].apply(dataset_evaluator.eval_row, axis=1)
        j=0
        while j<100:
          dataset_evaluator.multi_rows_feature_ex(df.iloc[j:(j+16)])
          j+=16
        dataset_evaluator.save_to_csv(f"{w}_data.csv")
        print(f"Saved {w} dataset!")

main()
