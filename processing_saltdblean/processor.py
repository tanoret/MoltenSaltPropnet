import re
import numpy as np
import pandas as pd
from typing import Dict
import warnings

class SALTDBLEANProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.predefined_elements = set()
        self.predefined_compounds = set()
        self._collect_predefined_components()

    def _collect_predefined_components(self):
        """Collect all elements and compounds present in the dataset."""
        for system in self.df['System']:
            compounds = [c.strip() for c in system.split('-')]
            for compound in compounds:
                self.predefined_compounds.add(compound)
                elements = self.parse_compound(compound)
                self.predefined_elements.update(elements.keys())

    @classmethod
    def from_csv(cls, path: str):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return cls(df)

    @staticmethod
    def non_zero(x):
        return x if x != 0 else 1e-12

    def parse_compound(self, c):
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            out[el] = out.get(el, 0) + int(n or "1")
        return out

    def compute_composition(self, row, composition_type='elements'):
        system = row['System']
        mol_frac = row['Mol Frac']
        compounds = system.split('-')

        # Determine fractions
        if mol_frac.strip() == 'Pure Salt':
            if len(compounds) != 1:
                raise ValueError("Pure Salt should have only one compound")
            fractions = [1.0]
        else:
            fractions = list(map(float, mol_frac.split('-')))
            if len(fractions) != len(compounds):
                raise ValueError("Number of fractions does not match number of compounds")

        # Compute compound compositions
        compound_dict = {compound: frac for compound, frac in zip(compounds, fractions)}

        # Compute elemental compositions
        total_composition = {}
        for compound, frac in zip(compounds, fractions):
            parsed_elements = self.parse_compound(compound)
            for element, count in parsed_elements.items():
                total_composition[element] = total_composition.get(element, 0) + frac * count
        total_sum = sum(total_composition.values())
        element_dict = {element: count / total_sum for element, count in total_composition.items()} if total_sum > 0 else {}

        # Return based on composition_type
        if composition_type == 'elements':
            return element_dict
        elif composition_type == 'compounds':
            return compound_dict
        elif composition_type == 'both':
            return {**element_dict, **compound_dict}
        else:
            raise ValueError("Invalid composition_type")

    def filter_by_components(self, filter_dict):
        """
        Filters the DataFrame to include only rows where the Composition contains specified
        elements/compounds. Additionally, includes elements from included compounds if they exist
        in the original composition. Updates the Composition column and removes rows with empty compositions.
        """
        include = filter_dict.get("include", {})
        elements_include = set(include.get("elements", []))
        compounds_include = set(include.get("compounds", []))

        # Create a copy of the DataFrame to avoid modifying the original
        filtered_df = self.df.copy()

        new_compositions = []
        mask = []

        for idx, row in filtered_df.iterrows():
            original_comp = row["Composition"]
            filtered_comp = {}

            # Step 1: Add explicitly included elements
            for el in elements_include:
                if el in original_comp:
                    filtered_comp[el] = original_comp[el]

            # Step 2: Add explicitly included compounds
            for cmp in compounds_include:
                if cmp in original_comp:
                    filtered_comp[cmp] = original_comp[cmp]

            # Step 3: Add elements from included compounds if they exist in the original composition
            for cmp in compounds_include:
                if cmp in original_comp:
                    parsed_elements = self.parse_compound(cmp).keys()
                    for el in parsed_elements:
                        if el in original_comp and el not in filtered_comp:
                            filtered_comp[el] = original_comp[el]

            new_compositions.append(filtered_comp)
            mask.append(bool(filtered_comp))

        # Update the DataFrame
        filtered_df["Composition"] = new_compositions
        filtered_df = filtered_df[mask].reset_index(drop=True)
        return SALTDBLEANProcessor(filtered_df)

    def compute_actual_properties(self, row, temperature):
        properties = {}
        try:
            if self.non_zero(row['rho_a']) or self.non_zero(row['rho_b']):
                prop = row['rho_a'] - row['rho_b'] * temperature
                if not np.isnan(prop):
                    properties['Density'] = prop
        except:
            pass
        try:
            if self.non_zero(row['mu1_a']) or self.non_zero(row['mu1_b']):
                prop = row['mu1_a'] * np.exp(row['mu1_b'] / (8.314 * temperature))
                if not np.isnan(prop):
                    properties['Viscosity A'] = prop
        except:
            pass
        try:
            if self.non_zero(row['mu2_a']) or self.non_zero(row['mu2_b']) or self.non_zero(row['mu2_c']):
                try:
                    prop = 10 ** (row['mu2_a'] + row['mu2_b'] / temperature + row['mu2_c'] / temperature**2)
                except RuntimeWarning as e:
                    print(f"Caught a RuntimeWarning: {e}")
                    prop = 1e-10
                warnings.resetwarnings()
                warnings.resetwarnings()
                if not np.isnan(prop):
                    properties['Viscosity B'] = prop
        except:
            pass
        try:
            if self.non_zero(row['k_a']) or self.non_zero(row['k_b']):
                prop = row['k_a'] + row['k_b'] * temperature
                if not np.isnan(prop):
                    properties['Thermal Conductivity'] = prop
        except:
            pass
        try:
            if self.non_zero(row['cp_a']) or self.non_zero(row['cp_b']) or self.non_zero(row['cp_c']) or self.non_zero(row['cp_d']):
                prop = row['cp_a'] + row['cp_b'] * temperature + row['cp_c'] / temperature**2 + row['cp_d'] * temperature**2
                if not np.isnan(prop):
                    properties['Heat Capacity of Liquid'] = prop
        except:
            pass
        return properties

    def compute_actual_properties_from_predictions(self, prediction, temperature):
        properties = {}
        R = 8.314

        rho_a = prediction.get('rho_a', 0.0)
        rho_b = prediction.get('rho_b', 0.0)
        prop = rho_a - rho_b * temperature
        if np.isfinite(prop):
            properties['Density'] = min(max(prop, 0.0), 1000.0)

        mu1_a = prediction.get('mu1_a', 0.0)
        mu1_b = prediction.get('mu1_b', 0.0)
        if mu1_a > 1e-10 and abs(mu1_b) > 1e-10:
            prop = mu1_a * np.exp(mu1_b / (R * temperature))
            properties['Viscosity A'] = min(max(prop, 0.0), 200.0)

        mu2_a = prediction.get('mu2_a', -100.0)
        mu2_b = prediction.get('mu2_b', 0.0)
        mu2_c = prediction.get('mu2_c', 0.0)
        if mu2_a > 1e-10 and abs(mu2_b) > 1e-10 and abs(mu2_c) > 1e-10:
            prop = 10 ** (mu2_a + mu2_b / temperature + mu2_c / temperature**2)
            properties['Viscosity B'] = min(max(prop, 0.0), 200.0)

        k_a = prediction.get('k_a', 0.0)
        k_b = prediction.get('k_b', 0.0)
        prop = k_a + k_b * temperature
        if np.isfinite(prop):
            properties['Thermal Conductivity'] = min(max(prop, 0.0), 100.0)

        cp_a = prediction.get('cp_a', 0.0)
        cp_b = prediction.get('cp_b', 0.0)
        cp_c = prediction.get('cp_c', 0.0)
        cp_d = prediction.get('cp_d', 0.0)
        try:
            prop = cp_a + cp_b * temperature + cp_c / temperature**2 + cp_d * temperature**2
            if np.isfinite(prop):
                properties['Heat Capacity of Liquid'] = min(max(prop, 0.0), 200.0)
        except Exception:
            properties['Heat Capacity of Liquid'] = 1e-10

        return properties

    def compute_properties(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        """Return actual properties at temperature *T* based on coefficient dict.
        Missing coefficients are replaced with zero."""
        R = 8.314
        p: Dict[str, float] = {}

        rho_a = coeffs.get("rho_a", 0.0)
        rho_b = coeffs.get("rho_b", 0.0)
        p["density"] = min(max(rho_a - rho_b * T, 0.0), 1000)

        mu1_a = coeffs.get("mu1_a", 0.0)
        mu1_b = coeffs.get("mu1_b", 0.0)
        if mu1_a > 1e-10 and abs(mu1_b) > 1e-10:
            p["viscosity_A"] = min(max(mu1_a * np.exp(mu1_b / (R * T)), 0), 200)

        mu2_a = coeffs.get("mu2_a", -100.0)
        mu2_b = coeffs.get("mu2_b", 0.0)
        mu2_c = coeffs.get("mu2_c", 0.0)
        if abs(mu2_a) > 1e-10 and abs(mu2_b) > 1e-10 and abs(mu2_c) > 1e-10:
            p["viscosity_B"] = min(max(10 ** (mu2_a + mu2_b / T + mu2_c / T**2), 0), 200)

        k_a = coeffs.get("k_a", 0.0)
        k_b = coeffs.get("k_b", 0.0)
        p["thermal_conductivity"] = min(max(k_a + k_b * T, 0), 100)

        cp_a = coeffs.get("cp_a", 0.0)
        cp_b = coeffs.get("cp_b", 0.0)
        cp_c = coeffs.get("cp_c", 0.0)
        cp_d = coeffs.get("cp_d", 0.0)
        try:
            p["heat_capacity"] = min(max(cp_a + cp_b * T + cp_c / T**2 + cp_d * T**2, 0), 200)
        except Exception:
            p["heat_capacity"] = 0.0

        return p

    def standardize_with_nan(self, data, mean, std):
        standardized = np.where(std > 0, (data - mean) / std, 0)
        return np.where(np.isnan(data), 0, standardized)

    def standardize(self, data, mean, std):
        return (data - mean) / std
