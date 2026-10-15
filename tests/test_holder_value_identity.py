"""Offline regression tests for holder-value protocol provenance."""
import unittest
import numpy as np
import pandas as pd
from data.holder_value_data import _candidate_matches, _evaluate_asset, _protocol_record

class HolderValueIdentityTests(unittest.TestCase):
    def table(self, names):
        return pd.DataFrame([_protocol_record({'name': n, 'slug': slug, 'total30d': 100}, 'dailyHoldersRevenue') for n, slug in names])

    def test_unique_name_identity(self):
        t=self.table([('Pump.fun','pump.fun')]); r=pd.Series({'name':'Pump.fun','coin_id':'pump.fun'})
        self.assertEqual(len(_candidate_matches(r,t)),1)

    def test_ambiguous_identity_fails_closed(self):
        t=self.table([('Pump.fun','pump.fun'),('Pump.fun','pumpswap')]); r=pd.Series({'name':'Pump.fun','coin_id':'pump.fun'})
        self.assertTrue(_candidate_matches(r,t).empty)

    def test_conflicting_identity_fails_closed(self):
        t=self.table([('Pump.fun','pump.fun'),('Other','other')]); r=pd.Series({'name':'Pump.fun','coin_id':'other'})
        self.assertTrue(_candidate_matches(r,t).empty)

    def test_no_source_not_marked_consistent(self):
        r=pd.Series({'name':'Unknown','coin_id':'unknown','market_cap':1e8})
        out=_evaluate_asset(r,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
        self.assertFalse(out['holder_value_cross_source_consistent'])
        self.assertFalse(out['holder_value_fee_ratio_verified'])
        self.assertTrue(np.isnan(out['fee_distribution_ratio']))

if __name__=='__main__':unittest.main()
