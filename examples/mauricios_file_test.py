import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_mstdb.processor    import MSTDBProcessor
from processing_mstdb.snn_trainer  import SNNMetaTrainer, TARGETS, DERIVED_PROPS






processor = MSTDBProcessor.from_csv("/Users/meggie/Documents/MoltenSaltPropnet/data/mstdb_processed.csv")
processor.df.columns = processor.df.columns.str.strip()

trainer = SNNMetaTrainer(processor.df, TARGETS, DERIVED_PROPS)

trainer.train_base()
trainer.train_meta()

# print rel-MSE (%) and R² for each target
trainer.evaluate()
