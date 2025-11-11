import pandas as pd
import meval  # pip install meval@git+https://github.com/FraunhoferMEVIS/meval


test_aligned_df = pd.read_csv("cxp_pneu_densenet_test_results_aligned.csv")
test_misaligned_df = pd.read_csv("cxp_pneu_densenet_test_results_misaligned.csv")

joint_df = pd.concat([test_aligned_df, test_misaligned_df], ignore_index=True)

meval.diags.roc_diag(joint_df, export_fig_size_cm=(12, 7), 
                     export_fig_path='roc.png', legend=True, 
                     fig_title=None,
                     plot_groups=['drain=0', 'drain=1'])
meval.diags.rel_diag(joint_df, export_fig_size_cm=(12, 7), 
                     export_fig_path='calibration.png', legend=True, 
                     fig_title=None,
                     plot_groups=['drain=0', 'drain=1'])