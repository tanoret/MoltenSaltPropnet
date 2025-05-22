import os
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor

def test_loading_and_composition():
    # Load sample data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mstdb_processed.csv")
    csv_path = os.path.abspath(csv_path)
    processor = MSTDBProcessor.from_csv(csv_path)

    # Check basic structure
    assert isinstance(processor.df, pd.DataFrame)

    # Check for an expected column (likely 'System')
    assert 'System' in [col.strip() for col in processor.df.columns]

    # Try computing composition using a cleaned row
    processor.df.columns = [col.strip() for col in processor.df.columns]
    row = processor.df.iloc[0]
    row = row.rename(lambda x: x.strip())  # remove any column whitespace
    row['formula'] = row['System']        # patch column for this test only
    comp = processor.compute_composition(row)
    assert isinstance(comp, dict)
    assert len(comp) > 0

def test_non_zero_behavior():
    assert MSTDBProcessor.non_zero(5) == 5
    assert MSTDBProcessor.non_zero(0) == 1e-12
