import json
import os
from eval import cal_em, cal_f1
from eval_r import cal_rsim
from eval_g import cal_gen
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
import os
import json
import traceback
from tqdm import tqdm
from eval import cal_em, cal_f1
from eval_r import cal_rsim
from eval_g import cal_gen
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

#  CLI 
parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='hypertension')
parser.add_argument('--part', default='')
parser.add_argument('--ts', default='', help='Run timestamp decided by input')
args = parser.parse_args()

data_source = args.data_source
part = args.part.strip()
part_tag = part if part else "orig"   

#  Result roots decided by (data_source, part) 
RUN_ROOT = f"results/{data_source}"
if part != "":
    RUN_ROOT = f"results/{data_source}_{part}"


#  RUN_TS / RUN_DIR: bash passes --ts; fallback to now() if missing 
RUN_TS = args.ts.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join(RUN_ROOT, RUN_TS)



def evaluate_one(d):
    try:
        generation = d['generation']
        try:
            answer = generation.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            answer = generation
        em_score = cal_em([d['golden_answers']], [answer])
        f1_score = cal_f1([d['golden_answers']], [answer])

        # dedup context
        context = []

        for c in d['context']:
            if c not in context:
                context.append(c)
        knowledge = ""
        if 'retrieved' in d:
            for item in d['retrieved']:
                knowledge += item['reasoning_path'] + "\n"
                knowledge += item['src_text_chunks'] + "\n"
                knowledge += item['entity_descriptions'] + "\n"
        rsim_score = cal_rsim(['\n'.join(context)], [knowledge]) if knowledge != "" else 0.0 


        gen_score = cal_gen(d['question'], d['golden_answers'], generation, f1_score)

        d['em'] = float(em_score)
        d['f1'] = float(f1_score)
        d['rsim'] = float(rsim_score)
        d['gen'] = float(gen_score["score"])
        d['gen_exp'] = gen_score["explanation"]

        return d
    except Exception as e:
        print(f"[ERROR] Failed processing sample: {d.get('question', 'N/A')}")
        traceback.print_exc()
        raise

def evaluate_method(args):

    data_source = args.data_source
    success_flag = False
    # part = args.part
    # if part != "":
    #     data_source = data_source + "_p"

    try:
        print(f"[DEBUG] Evaluating  on {data_source}")

        answer_file = os.path.join(RUN_DIR, "generated_answer.json")
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"File not found: {answer_file}")
        

        with open(answer_file) as f:
            data = json.load(f)

        # 并行处理样本
        max_workers =16
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            method = "pokh"
            data = list(tqdm(executor.map(evaluate_one, data), total=len(data), desc=method))

        # 汇总指标
        overall_em = sum([d['em'] for d in data]) / len(data)
        overall_f1 = sum([d['f1'] for d in data]) / len(data)
        overall_rsim = sum([d['rsim'] for d in data]) / len(data)
        overall_gen = sum([d['gen'] for d in data]) / len(data)

        print(f" Overall EM: {overall_em:.4f}")
        print(f" Overall F1: {overall_f1:.4f}")
        print(f" Overall R-Sim: {overall_rsim:.4f}")
        print(f" Overall Gen: {overall_gen:.4f}")



        result_path = os.path.join(RUN_DIR, "test_result.json")
        with open(result_path, 'w') as f:
            json.dump(data, f, indent=4)

        score_path = os.path.join(RUN_DIR, "test_score.json")
        with open(score_path, 'w') as f:
            json.dump({
                "overall_em": overall_em,
                "overall_f1": overall_f1,
                "overall_rsim": overall_rsim,
                "overall_gen": overall_gen,
            }, f, indent=4)

        # 成功保存标志
        success_flag = True
        print(f"[SAVED] {result_path}")
        print(f"[SAVED] {score_path}")
        print(f"[SUCCESS]  finished and saved.")

    except Exception as e:
        print(f"\n[ERROR]  failed due to: {str(e)}")
        traceback.print_exc()
        raise

    if not success_flag:
        raise RuntimeError(f" did not complete saving.")
    
    return True

if __name__ == "__main__":

    evaluate_method(args)