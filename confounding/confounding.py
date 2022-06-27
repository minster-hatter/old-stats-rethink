from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork
from pgmpy.inference import CausalInference

# Backdoor paths from section 6.4.2.
dag_642 = DAG([("A", "C"), ("C", "B"), ("C", "Y"), ("X", "Y")])
dag_642.add_node("(u)", latent=True)
dag_642.add_edge("(u)", "B")
dag_642.add_edge("(u)", "X")
dag_642.add_edge("A", "(u)")
dag_642_plot = dag_642.to_daft(
    # node_pos="circular",
    pgm_params={"observed_style": "inner"}
)
dag_642_plot.render()
dag_642_plot.savefig("dag_642.png")

with open("dag_642_conditional_independencies.txt", "w") as output:
    output.write(
        f"DAG_642 conditional independencies:\n{dag_642.get_independencies()}\n"
    )

bn_642 = BayesianNetwork(dag_642)
inference_642 = CausalInference(bn_642)

with open("dag_642_XY_backdoor_adjustments.txt", "w") as output:
    output.write(str(inference_642.get_all_backdoor_adjustment_sets("X", "Y")))

with open("dag_642_XY_minimal_adjustment.txt", "w") as output:
    output.write(str(inference_642.get_minimal_adjustment_set("X", "Y")))

# Backdoor paths from section 6.4.3.
dag_643 = DAG(
    [
        ("S", "W"),
        ("S", "M"),
        ("S", "A"),
        ("W", "D"),
        ("M", "D"),
        ("A", "D"),
        ("A", "M"),
        ("M", "D"),
    ]
)
dag_643_plot = dag_643.to_daft(pgm_params={"observed_style": "inner"})
dag_643_plot.render()
dag_643_plot.savefig("dag_643.png")

with open("dag_643_conditional_independencies.txt", "w") as output:
    output.write(
        f"DAG_643 conditional independencies:\n{dag_643.get_independencies()}\n"
    )

bn_643 = BayesianNetwork(dag_643)
inference_643 = CausalInference(bn_643)

print(inference_643.get_all_backdoor_adjustment_sets("W", "D"))
print(inference_643.get_minimal_adjustment_set("W", "D"))

with open("dag_643_WD_backdoor_adjustments.txt", "w") as output:
    output.write(str(inference_643.get_all_backdoor_adjustment_sets("W", "D")))

with open("dag_643_WD_minimal_adjustment.txt", "w") as output:
    output.write(str(inference_643.get_minimal_adjustment_set("W", "D")))
