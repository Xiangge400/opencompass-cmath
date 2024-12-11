from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import CMath
from opencompass.utils.text_postprocessors import first_capital_postprocess

cmath_datasets = []
for split in ['test', 'validation']:
    cmath_reader_cfg = dict(
        input_columns=['question', 'grade', 'reasoning_step', 'num_digits'],
        output_column='golden',
        train_split=split,
        test_split=split
    )

    cmath_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round = [
                    dict(
                        role='HUMAN',
                        prompt=f'以下是一道{{grade}}年级的数学题，不需要任何分析，解释，直接以一个数字的形式输出答案。 \n{{question}} \n 答案：',
                    ),
                    dict(role='BOT', prompt='{golden}'),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=10),
    )

    cmath_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess),
    )

    cmath_datasets.append(
        dict(
            abbr='cmath',
            type=CMath,
            path='./data/CMath/',
            reader_cfg=cmath_reader_cfg,
            infer_cfg=cmath_infer_cfg,
            eval_cfg=cmath_eval_cfg,
        )
    )