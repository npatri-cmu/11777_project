from src.evaluation import metrics, paired_interval, category

def test_denominators():
    rows=[dict(pair_id=0,image_condition='counterfactual',prediction='red',typical_labels=['red'],correct=False,baseline_correct=True),
          dict(pair_id=1,image_condition='ordinary',prediction='green',typical_labels=['green'],correct=True,baseline_correct=False)]
    m=metrics(rows)
    assert m['prior_consistent_error_rate']==1 and m['correction_rate']==1 and m['regression_rate']==1
    assert paired_interval(rows)['delta']==0
    assert category(False,False)=='both_incorrect'
    assert metrics([])['correction_rate'] is None
