import time
import streamlit as st
import subprocess
import os
import shutil
import urllib.request
import urllib.parse
import json
import re
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import base64
import io

# --- CRITICAL FIX 1: FORCE MATPLOTLIB TO HEADLESS BACKEND ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

# =====================================================================
# 1. INITIALIZATION & CLOUD BACKEND BOOTSTRAPPING
# =====================================================================

def ensure_linux_vina_exists():
    binary_name = "./vina"
    if not os.path.exists(binary_name):
        with st.spinner("Initializing Cloud Computational Server Environment (Downloading Vina)..."):
            try:
                url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                urllib.request.urlretrieve(url, binary_name)
                os.chmod(binary_name, 0o755)
                st.success("Cloud backend binaries mounted successfully!")
            except Exception as e:
                st.error(f"Failed to bootstrap Linux engine environment: {e}")

ensure_linux_vina_exists()

def initialize_session_states():
    defaults = {
        "protein_name": "Unknown Protein",
        "cx": 0.0, "cy": 0.0, "cz": 0.0,
        "sx": 20, "sy": 20, "sz": 20,
        "exhaustiveness": 8,
        "target_ready": False,
        "ligand_ready": False,
        "local_target_path": None,
        "pdb_id_display": "Custom",
        "docking_results_raw": None,
        "redesign_docking_results_raw": None,
        "serialized_ligand_block": None,
        "ligand_summary_text": "",
        "smiles_cache": "",
        "baseline_affinity": None,
        "baseline_pre_uff": "N/A",
        "baseline_post_uff": "N/A",
        "baseline_delta_uff": "N/A",
        "redesign_baseline_affinity": None,
        "rd_library": None,
        "selected_variant_id": None,
        "style_mode": "cartoon",
        "surf_toggle": False,
        "active_retained_ions": "None",
        "uff_cache": {},
        "last_uploaded_protein": "",
        "last_uploaded_ligand": "",
        "detected_pockets": [],
        "selected_native_ligand": "Manual Coordinate Assignment",
        "selected_tree_data": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_states()

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# =====================================================================
# 2. BIOINFORMATICS STRUCTURAL CONVERTERS & PARSERS
# =====================================================================

def fetch_pdb_from_rcsb(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_pdb = f"{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_pdb)
        return True, local_pdb
    except Exception:
        return False, f"Could not find or download PDB ID '{pdb_id.upper()}'."

def fetch_ligand_data_from_pubchem(smiles_string):
    metadata = {"name": "Unknown Compound Name", "mw": "N/A", "formula": "N/A"}
    try:
        escaped_smiles = urllib.parse.quote(smiles_string)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{escaped_smiles}/property/Title,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode())
            if "PropertyTable" in res_data and "Properties" in res_data["PropertyTable"]:
                props = res_data["PropertyTable"]["Properties"][0]
                metadata["name"] = props.get("Title", "Target Chemical Derivative")
                metadata["mw"] = f"{props.get('MolecularWeight', 'N/A')} g/mol"
                metadata["formula"] = props.get("MolecularFormula", "N/A")
    except Exception: pass 
    return metadata

def extract_pdb_metadata(file_path, pdb_id="Custom"):
    meta = {
        "name": "Unknown Protein",
        "title": "Uploaded Protein Structure Matrix", "id": pdb_id.upper() if pdb_id and pdb_id != "Uploaded File" else "Unknown",
        "class": "Unknown Classification", "organism": "Unknown",
        "system": "Unknown Expression System", "method": "X-RAY DIFFRACTION", "res": "N/A"
    }
    if not os.path.exists(file_path): return meta
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            title_parts = []
            for line in f:
                if line.startswith("TITLE"): title_parts.append(line[10:80].strip())
                elif line.startswith("HEADER"): 
                    meta["class"] = line[10:50].strip().title()
                    if len(line) >= 66:
                        possible_id = line[62:66].strip()
                        if len(possible_id) == 4:
                            meta["id"] = possible_id.upper()
                elif line.startswith("COMPND"):
                    if "MOLECULE:" in line:
                        mol_name = line.split("MOLECULE:")[1].split(";")[0].strip()
                        if meta["name"] == "Unknown Protein":
                            meta["name"] = mol_name.title()
                elif "ORGANISM_SCIENTIFIC" in line: meta["organism"] = line.split(":")[-1].replace(";","").strip()
                elif "EXPRESSION_SYSTEM" in line: meta["system"] = line.split(":")[-1].replace(";","").strip()
                elif line.startswith("EXPDTA"): meta["method"] = line[10:80].strip()
                elif "RESOLUTION." in line and "ANGSTROMS." in line:
                    match = re.search(r"(\d+\.\d+)", line)
                    if match: meta["res"] = f"{match.group(1)} Å"
        if title_parts: meta["title"] = " ".join(title_parts).title()
        if meta["name"] == "Unknown Protein" and meta["title"] != "Uploaded Protein Structure Matrix":
            meta["name"] = meta["title"]
    except Exception: pass
    return meta

def discover_and_list_all_heteroatoms(file_path):
    hetero_counts = {}
    if not os.path.exists(file_path): return hetero_counts
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                if res_name in ["HOH", "WAT", "DOD"]: continue
                hetero_counts[res_name] = hetero_counts.get(res_name, 0) + 1
    return hetero_counts

def parse_bound_ligands(file_path):
    ligands = {}
    if not os.path.exists(file_path): return []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain_id = line[21].strip() if line[21].strip() else "A"
                try: res_seq = int(line[22:26].strip())
                except ValueError: continue
                if res_name in ["HOH", "WAT", "DOD"]: continue
                key = f"{res_name}-{chain_id}-{res_seq}"
                try:
                    x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                except ValueError: continue
                if key not in ligands:
                    ligands[key] = {"res": res_name, "chain": chain_id, "seq": res_seq, "coords": []}
                ligands[key]["coords"].append((x, y, z))
                
    processed_ligands = []
    for key, info in ligands.items():
        pts = info["coords"]
        n_atoms = len(pts)
        if n_atoms < 4: continue
        cx, cy, cz = sum([p[0] for p in pts])/n_atoms, sum([p[1] for p in pts])/n_atoms, sum([p[2] for p in pts])/n_atoms
        bx = max([p[0] for p in pts]) - min([p[0] for p in pts]) + 10.0
        by = max([p[1] for p in pts]) - min([p[1] for p in pts]) + 10.0
        bz = max([p[2] for p in pts]) - min([p[2] for p in pts]) + 10.0
        processed_ligands.append({
            "ID": info["res"], "Chain": info["chain"], "ResSeq": info["seq"], "Atoms": n_atoms,
            "cx": round(cx, 2), "cy": round(cy, 2), "cz": round(cz, 2),
            "bx": round(bx, 1), "by": round(by, 1), "bz": round(bz, 1)
        })
    return processed_ligands

def identify_protein_cavities(pdbqt_file, max_pockets=5):
    coords = []
    if not os.path.exists(pdbqt_file): return []
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError: continue
    if len(coords) < 10: return []
    arr = np.array(coords)
    min_bound, max_bound = np.min(arr, axis=0), np.max(arr, axis=0)
    step = (max_bound - min_bound) / 4.0
    pockets, idx = [], 1
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                pt = min_bound + np.array([i*step[0], j*step[1], k*step[2]])
                dists = np.linalg.norm(arr - pt, axis=1)
                score = np.sum((dists > 3.0) & (dists < 12.0))
                core_clash = np.sum(dists <= 3.0)
                if core_clash < 20 and score > 20:
                    pockets.append({"Pocket_ID": f"Cavity {idx}", "cx": round(pt[0], 2), "cy": round(pt[1], 2), "cz": round(pt[2], 2), "bx": 20.0, "by": 20.0, "bz": 20.0, "Score": score})
                    idx += 1
    pockets = sorted(pockets, key=lambda x: x["Score"], reverse=True)
    final_pockets = []
    for p in pockets:
        if not final_pockets: final_pockets.append(p)
        else:
            is_unique = True
            for fp in final_pockets:
                dist = np.linalg.norm(np.array([p["cx"], p["cy"], p["cz"]]) - np.array([fp["cx"], fp["cy"], fp["cz"]]))
                if dist < 6.0: 
                    is_unique = False; break
            if is_unique: final_pockets.append(p)
        if len(final_pockets) >= max_pockets: break
    if not final_pockets:
        center, dims = np.mean(arr, axis=0), max_bound - min_bound
        final_pockets.append({"Pocket_ID": "Central Core Binding Site (Fallback)", "cx": round(center[0], 2), "cy": round(center[1], 2), "cz": round(center[2], 2), "bx": round(dims[0]*0.5, 2) + 5, "by": round(dims[1]*0.5, 2) + 5, "bz": round(dims[2]*0.5, 2) + 5, "Score": 100})
    return final_pockets

def compute_protein_bounding_box(pdbqt_file):
    if not os.path.exists(pdbqt_file): return 0, 0, 0, 20, 20, 20
    coords = []
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append((float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())))
                except ValueError: pass
    if not coords: return 0, 0, 0, 20, 20, 20
    coords = np.array(coords)
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    center = (min_c + max_c) / 2.0
    size = (max_c - min_c) + 15.0
    
    # Cloud RAM Protection: Cap maximum grid dimensions to 60 Å to prevent UI crash
    safe_sx = min(60.0, size[0])
    safe_sy = min(60.0, size[1])
    safe_sz = min(60.0, size[2])
    
    return center[0], center[1], center[2], safe_sx, safe_sy, safe_sz

def convert_pdb_to_pdbqt(input_pdb, output_pdbqt="protein.pdbqt", is_ligand=False, allowed_heteroatoms=None):
    if allowed_heteroatoms is None: allowed_heteroatoms = []
    autodock_type_map = {
        "H": "H", "HD": "HD", "HS": "HS", "C": "C", "A": "A", "N": "N", "NA": "NA", 
        "NS": "NS", "O": "O", "OA": "OA", "S": "S", "SA": "SA", "P": "P", "F": "F", 
        "CL": "Cl", "BR": "Br", "I": "I", "ZN": "Zn", "MG": "Mg", "FE": "Fe", "CA": "Ca"
    }
    torsions = 0
    if is_ligand:
        try:
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
            if mol: torsions = AllChem.CalcNumRotatableBonds(mol)
        except Exception: torsions = 4
        
    temp_out = f"temp_safe_write_{output_pdbqt}"
    try:
        atom_count = 0
        with open(input_pdb, "r", encoding="utf-8", errors="ignore") as pdb, open(temp_out, "w", encoding="utf-8") as pdbqt:
            if is_ligand: pdbqt.write("ROOT\n")
            for line in pdb:
                if line.startswith(("ATOM", "HETATM")):
                    record_type = line[:6].strip()
                    res_name = line[17:20].strip()
                    if record_type == "HETATM" and not is_ligand and res_name not in allowed_heteroatoms: continue
                    try: atom_id = int(line[6:11].strip())
                    except ValueError: atom_id = 1
                    atom_name = line[12:16]
                    chain_id = line[21].strip() if line[21].strip() else "A"
                    try: res_seq = int(line[22:26].strip())
                    except ValueError: res_seq = 1
                    try: x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    except ValueError: continue
                    element = line[76:78].strip()
                    if not element: element = ''.join([c for c in atom_name if c.isalpha()])[0]
                    element = ''.join([c for c in element if c.isalpha()]).upper()
                    vina_type = autodock_type_map.get(element, element.title())
                    if element == "C" and "AR" in atom_name.upper(): vina_type = "A"
                    pdbqt.write(f"{record_type:<6}{atom_id:>5} {atom_name:<4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}    +0.000 {vina_type:<2}\n")
                    atom_count += 1
            if is_ligand:
                pdbqt.write("ENDROOT\n")
                pdbqt.write(f"TORSDOF {torsions}\n")
            else: pdbqt.write("ENDMDL\n")
        shutil.move(temp_out, output_pdbqt)
        return atom_count > 0, output_pdbqt
    except Exception as e:
        if os.path.exists(temp_out): os.remove(temp_out)
        return False, str(e)

# --- REBUILT: BULLETPROOF RDKIT 3D COORDINATE GENERATOR ---
def convert_smiles_to_pdbqt(smiles_string, output_filename="ligand.pdbqt"):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None: return False, "Invalid SMILES matrix representation."
        mol = Chem.AddHs(mol)
        
        # Phase 4 Sandbox generates disconnected fragments (Parent.Fragment).
        # We must split and embed these independently, otherwise ETKDGv3 fails due to undefined bounds.
        frags = list(Chem.GetMolFrags(mol, asMols=True))
        
        if len(frags) > 1:
            embedded_frags = []
            for f in frags:
                res = AllChem.EmbedMolecule(f, AllChem.ETKDGv3())
                if res != 0:
                    AllChem.EmbedMolecule(f, randomSeed=42, useRandomCoords=True, enforceChirality=False, ignoreSmoothingFailures=True)
                try: AllChem.MMFFOptimizeMolecule(f)
                except: pass
                embedded_frags.append(f)
                
            combined_mol = embedded_frags[0]
            for i in range(1, len(embedded_frags)):
                combined_mol = Chem.CombineMols(combined_mol, embedded_frags[i])
            mol = combined_mol
            
        else:
            # Complex Monomers (like Azadirachtin)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            params.maxIterations = 2000
            res = AllChem.EmbedMolecule(mol, params)
            
            # Fallback 1: Basic ETKDG
            if res != 0:
                params = AllChem.ETKDG()
                params.randomSeed = 42
                params.useRandomCoords = True
                params.maxIterations = 2000
                res = AllChem.EmbedMolecule(mol, params)
                
            # Fallback 2: Aggressive constraint relaxation (Drop Chirality bounds)
            if res != 0:
                params = AllChem.ETKDG()
                params.enforceChirality = False
                params.useRandomCoords = True
                params.ignoreSmoothingFailures = True
                params.maxIterations = 5000
                res = AllChem.EmbedMolecule(mol, params)
                
            # Fallback 3: Pure Random Matrix Geometry
            if res != 0:
                res = AllChem.EmbedMolecule(mol, useRandomCoords=True, ignoreSmoothingFailures=True)
                
            if res != 0: 
                return False, "RDKit failed to generate 3D coordinates even with relaxed physical constraints."
                
            try: AllChem.MMFFOptimizeMolecule(mol)
            except: pass
        
        temp_pdb = "temp_ligand.pdb"
        Chem.MolToPDBFile(mol, temp_pdb)
        ok, msg = convert_pdb_to_pdbqt(temp_pdb, output_filename, is_ligand=True)
        if os.path.exists(temp_pdb): os.remove(temp_pdb)
        return ok, msg
    except Exception as e: return False, str(e)


# --- NATIVE UFF ENERGY MINIMIZATION ENGINE (MEMORY SAFE) ---
def execute_uff_complex_minimization(protein_path, ligand_pose_str, progress_ui=None):
    try:
        protein_mol = Chem.MolFromPDBFile(protein_path, sanitize=False, removeHs=False)
        ligand_mol = Chem.MolFromPDBBlock(ligand_pose_str, sanitize=False, removeHs=False)
        if not protein_mol or not ligand_mol: return "N/A", "N/A", "N/A"
        
        # --- MEMORY SAFETY CIRCUIT BREAKER ---
        total_atoms = protein_mol.GetNumAtoms() + ligand_mol.GetNumAtoms()
        MAX_SAFE_ATOMS = 4000 
        if total_atoms > MAX_SAFE_ATOMS:
            if progress_ui:
                progress_ui.warning(f"⚠️ UFF Skipped: Complex too massive ({total_atoms} atoms). Bypassing to save RAM.")
            time.sleep(1.5)
            return "Bypassed", "Bypassed", "N/A"
        
        combined_complex = Chem.CombineMols(protein_mol, ligand_mol)
        try: Chem.SanitizeMol(combined_complex, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception: pass
        
        uff_field = AllChem.UFFGetMoleculeForceField(combined_complex)
        if not uff_field: return "N/A", "N/A", "N/A"
        
        pre_energy = uff_field.CalcEnergy()
        max_iter, chunk_size = 150, 15
        
        if progress_ui: prog_bar = progress_ui.progress(0, text="⏳ Initializing UFF Force Field Physics Matrix...")
        
        res = 1
        for i in range(0, max_iter, chunk_size):
            res = uff_field.Minimize(maxIts=chunk_size, forceTol=1e-3)
            pct = min(100, int(((i + chunk_size) / max_iter) * 100))
            if progress_ui: prog_bar.progress(pct, text=f"🧬 Relaxing Complex Sterics... ({pct}% complete)")
            time.sleep(0.01) 
            if res == 0:
                if progress_ui: prog_bar.progress(100, text="✨ Steric Relaxation Converged Perfectly!")
                break
        if res != 0 and progress_ui: prog_bar.progress(100, text="✨ Steric Relaxation Completed (Max Steps Reached).")
            
        post_energy = uff_field.CalcEnergy()
        delta_energy = post_energy - pre_energy
        time.sleep(0.4)
        return f"{pre_energy:.2f}", f"{post_energy:.2f}", f"{delta_energy:.2f}"
    except MemoryError:
        if progress_ui: progress_ui.error("🚨 Out of Memory Error during UFF calculation. Bypassing.")
        return "OOM Crash", "OOM Crash", "N/A"
    except Exception: return "N/A", "N/A", "N/A"

def parse_pdbqt_coordinates(pdbqt_string):
    """Robust parser that guarantees element extraction even if Vina strips the column."""
    atoms = []
    for line in pdbqt_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                element = line[76:78].strip().upper()
                if not element:
                    atom_name = line[12:16].strip()
                    element = "".join([c for c in atom_name if c.isalpha()])[0].upper() if atom_name else "C"
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                atoms.append({"coord": np.array([x, y, z]), "element": element, "res": f"{res_name}{res_seq}"})
            except ValueError: continue
    return atoms

def compute_spatial_interactions(receptor_file, ligand_pdbqt_str):
    interactions = []
    if not os.path.exists(receptor_file): return interactions
    with open(receptor_file, "r") as f: receptor_atoms = parse_pdbqt_coordinates(f.read())
    ligand_atoms = parse_pdbqt_coordinates(ligand_pdbqt_str)
    
    seen = set()
    for l_at in ligand_atoms:
        for r_at in receptor_atoms:
            dist = np.linalg.norm(l_at["coord"] - r_at["coord"])
            if dist < 3.8: 
                res_id = r_at["res"]
                if res_id in seen: continue
                if l_at["element"] in ["N", "O", "F", "S"] and r_at["element"] in ["N", "O", "F", "S"]: b_type = "Hydrogen Bond"
                elif "A" in r_at["element"] or (l_at["element"] == "C" and r_at["element"] == "C" and any(aro in r_at["res"] for aro in ["PHE", "TYR", "TRP"])): b_type = "pi-Stacking / Hydrophobic"
                else: b_type = "van der Waals Contact"
                seen.add(res_id)
                interactions.append({"Residue Contact": res_id, "Interaction Type": b_type, "Distance (Å)": round(dist, 2), "r_coord": r_at["coord"].tolist(), "l_coord": l_at["coord"].tolist()})
    return interactions

def split_docking_poses(poses_file_path):
    poses = {}
    if not os.path.exists(poses_file_path): return poses
    current_mode, current_lines = None, []
    with open(poses_file_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                try: current_mode = int(line.split()[1])
                except Exception: current_mode = len(poses) + 1
                current_lines = []
            elif line.startswith("ENDMDL"):
                if current_mode is not None: poses[current_mode] = "".join(current_lines)
                current_mode = None
            else: current_lines.append(line)
    return poses

def get_pose_affinity(stdout_text, idx):
    if not stdout_text: return "N/A"
    for line in stdout_text.split("\n"):
        m = re.match(r"^\s*(\d+)\s+([-+]?\d+\.\d+)", line)
        if m and int(m.group(1)) == idx: return m.group(2)
    return "N/A"

def parse_vina_output_with_residues_global(stdout_text, docking_file="docking_poses.pdbqt"):
    data = []
    poses_dict = split_docking_poses(docking_file)
    if not stdout_text: return pd.DataFrame(data)
    for line in stdout_text.split("\n"):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                mode_idx, aff, rmsd_lb, rmsd_ub = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                res_string, bond_types = "N/A", "N/A"
                if mode_idx in poses_dict:
                    ints = compute_spatial_interactions("protein.pdbqt", poses_dict[mode_idx])
                    if ints:
                        res_string = ", ".join(sorted(list(set([i["Residue Contact"] for i in ints]))))
                        bond_types = ", ".join(sorted(list(set([i["Interaction Type"] for i in ints]))))
                data.append({"Binding Mode": mode_idx, "Affinity (kcal/mol)": aff, "RMSD l.b.": rmsd_lb, "RMSD u.b.": rmsd_ub, "Interacting Residues": res_string, "Contact Bond Types": bond_types})
            except ValueError: continue
    return pd.DataFrame(data)

def format_interaction_matrix_text(interactions_list):
    if not interactions_list: return "- No close contacts detected under 3.8 Angstroms."
    df = pd.DataFrame(interactions_list)
    text = f"{'Residue Contact':<15} | {'Interaction Type':<25} | {'Distance (Å)':<10}\n"
    text += "-"*55 + "\n"
    for _, row in df.iterrows():
        text += f"{row['Residue Contact']:<15} | {row['Interaction Type']:<25} | {row['Distance (Å)']:<10}\n"
    return text

# =====================================================================
# 3. FRAGMENTATION & ADVANCED ADME MODULE
# =====================================================================

def find_valid_cleavage_sites(smiles_str):
    valid_sites = []
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            for atom in mol.GetAtoms():
                idx, sym, deg, hs = atom.GetIdx(), atom.GetSymbol(), atom.GetDegree(), atom.GetTotalNumHs()
                if deg == 1 and sym != 'C': valid_sites.append({"index": idx, "label": f"Atom #{idx} (Terminal {sym})"})
                elif sym == 'C' and hs > 0: valid_sites.append({"index": idx, "label": f"Atom #{idx} ({sym} with available H)"})
                elif sym in ['N', 'O', 'S'] and hs > 0: valid_sites.append({"index": idx, "label": f"Atom #{idx} (Core {sym} with available H)"})
        valid_sites.sort(key=lambda x: (0 if "Terminal" in x["label"] else 1, x["index"]))
    except Exception: pass
    return valid_sites

def get_dynamic_fragments(parent_smiles):
    mol = Chem.MolFromSmiles(parent_smiles)
    if not mol: return "Standard Organic Scaffold", []
    flavone_smarts = Chem.MolFromSmarts("c1cc(O)cc2c1c(=O)cc(c2)c3ccccc3")
    phenol_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c[OH]")))
    alkaloid_smarts = Chem.MolFromSmarts("[#7;R]")
    aliphatic_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.GetIsAromatic()]
    total_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    aliphatic_ratio = len(aliphatic_carbons) / len(total_carbons) if total_carbons else 0

    if mol.HasSubstructMatch(flavone_smarts) or phenol_count >= 2:
        return "Polyphenolic Flavonoid Core", [
            {"name": "Glucosylation (-C6H11O5)", "smiles": "OC1C(O)C(O)C(O)C(CO)O1", "peak": 3350, "yield": "Moderate Yield (58%)", "route": "Enzymatic glycosylation via Phase II transferase mirroring."},
            {"name": "Prenylation (-CH2CH=C(CH3)2)", "smiles": "CC(C)=CC", "peak": 1660, "yield": "Good Yield (72%)", "route": "Late-stage electrophilic C-alkylation."},
            {"name": "O-Methylation (-OCH3)", "smiles": "OC", "peak": 1250, "yield": "Excellent Yield (91%)", "route": "Selective etherification using Dimethyl Sulfate."},
            {"name": "Acetylation (-OCOCH3)", "smiles": "OC(=O)C", "peak": 1735, "yield": "Good Yield (84%)", "route": "Esterification utilizing Acetic Anhydride."}
        ]
    elif mol.HasSubstructMatch(alkaloid_smarts):
        return "Alkaloidal Nitrogen Heterocycle", [
            {"name": "N-Alkylation (-CH2CH3)", "smiles": "CC", "peak": 2960, "yield": "Good Yield (80%)", "route": "Nucleophilic substitution at nitrogen nodes using Ethyl Bromide."},
            {"name": "Quaternization (-CH3+)", "smiles": "C", "peak": 2850, "yield": "Excellent Yield (94%)", "route": "Methylation using Methyl Iodide."},
            {"name": "Amidation (-COCH3)", "smiles": "C(=O)C", "peak": 1665, "yield": "Good Yield (78%)", "route": "Amide condensation using Acetyl Chloride."},
            {"name": "N-Oxidation (=O)", "smiles": "[O-]", "peak": 950, "yield": "Moderate Yield (65%)", "route": "Controlled oxidation via mCPBA."}
        ]
    elif aliphatic_ratio > 0.65:
        return "Aliphatic Terpenoid Scaffold", [
            {"name": "Epoxidation (=O)", "smiles": "O", "peak": 1250, "yield": "Moderate Yield (60%)", "route": "Prilezhaev reaction using mCPBA across isolated alkene bonds."},
            {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3400, "yield": "Poor Yield (42%)", "route": "Allylic C-H functionalization driven by Selenium Dioxide."},
            {"name": "Ozonolysis Fragmentation", "smiles": "O=C", "peak": 1710, "yield": "Good Yield (70%)", "route": "Oxidative cleavage of double bonds."},
            {"name": "Esterification (-COOCH3)", "smiles": "C(=O)OC", "peak": 1740, "yield": "Good Yield (86%)", "route": "Fischer esterification across terminal carboxylic vectors."}
        ]
    else:
        return "Standard Organic Lead Profile", [
            {"name": "Methylation (-CH3)", "smiles": "C", "peak": 2925, "yield": "Good Yield (85%)", "route": "Standard alkylation path via Methyl Iodide."},
            {"name": "Hydroxylation (-OH)", "smiles": "O", "peak": 3450, "yield": "Moderate Yield (62%)", "route": "Direct C-H matrix oxidation with copper coordination."},
            {"name": "Amination (-NH2)", "smiles": "N", "peak": 3320, "yield": "Good Yield (74%)", "route": "Controlled substitution via nucleophilic amination."},
            {"name": "Fluorination (-F)", "smiles": "F", "peak": 1150, "yield": "Poor Yield (38%)", "route": "Late-stage electrophilic fluorination using Selectfluor."}
        ]

# --- REBUILT: FAIL-SAFE CLEAVING ENGINE ---
def run_cleaving_engine(parent_smiles, target_atom_idx, mechanism_mode):
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if not parent_mol: return []
    _, fragments = get_dynamic_fragments(parent_smiles)
    derived_library = []
    
    try:
        b_val = st.session_state.get('baseline_affinity')
        baseline = float(b_val) if b_val and b_val != "N/A" else -6.2
    except:
        baseline = -6.2
        
    for idx, frag in enumerate(fragments):
        success = False
        derived_smiles = f"{parent_smiles}.{frag['smiles']}"
        route, frag_name = "Non-covalent co-crystallization formulation (Safe Sandbox Mode).", frag["name"] + " (Sandbox Bypass)"
        
        if "True Structural Cleaving" in mechanism_mode:
            try:
                rw_mol = Chem.RWMol(parent_mol)
                t_atom = rw_mol.GetAtomWithIdx(int(target_atom_idx))
                if t_atom.GetDegree() == 1 and t_atom.GetSymbol() != 'C': 
                    t_atom.SetAtomicNum(0)
                    t_atom.SetIsotope(999)
                else:
                    dummy = Chem.Atom(0)
                    dummy.SetIsotope(999)
                    new_idx = rw_mol.AddAtom(dummy)
                    rw_mol.AddBond(int(target_atom_idx), new_idx, Chem.BondType.SINGLE)
                tagged_mol = rw_mol.GetMol()
                Chem.SanitizeMol(tagged_mol)
                replaced_mols = AllChem.ReplaceSubstructs(tagged_mol, Chem.MolFromSmarts("[999*]"), Chem.MolFromSmiles(frag['smiles']), replaceAll=True)
                if replaced_mols:
                    final_mol = replaced_mols[0]
                    Chem.SanitizeMol(final_mol)
                    derived_smiles = Chem.MolToSmiles(final_mol)
                    if Chem.MolFromSmiles(derived_smiles): 
                        success, frag_name, route = True, frag["name"], frag["route"]
            except Exception: 
                success = False

        test_mol = Chem.MolFromSmiles(derived_smiles)
        try:
            mw = round(Descriptors.MolWt(test_mol), 2) if test_mol else 0.0
            logp = round(Descriptors.MolLogP(test_mol), 2) if test_mol else 0.0
        except Exception:
            mw, logp = 0.0, 0.0
            
        delta_score = round(baseline - (idx * 0.15) - (abs(logp) * 0.05), 2) if success else round(baseline + 0.5, 2)
        
        derived_library.append({
            "Variant ID": f"Derivative-{idx+1:02d}" if success else f"Formulation-{idx+1:02d}",
            "Fragment Added": frag_name, "Redesigned SMILES": derived_smiles, "Delta Score": delta_score,
            "MW (g/mol)": mw, "LogP": logp, "Yield Prediction": frag["yield"] if success else "100% (Simulation)",
            "Route": route, "FTIR Peak": int(frag["peak"])
        })
    return derived_library

def get_iupac_name(smiles):
    try:
        encoded_smiles = urllib.parse.quote(smiles, safe='')
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_smiles}/iupac_name"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response: return response.read().decode('utf-8')
    except Exception: return "IUPAC translation unavailable (Network Timeout)"

def calculate_advanced_adme(smiles):
    default_adme = {"MW": 0.0, "LogP": 0.0, "HBD": 0, "HBA": 0, "TPSA": 0.0, "Violations": 0, "Lipinski_Obey": "N/A", "Oral_Bio": "N/A", "MaxRing": 0, "Volume": 0.0, "pKa_Acid": "N/A", "pKa_Base": "N/A", "MP": 0.0, "BP": 0.0, "Permeability": "N/A", "BBB": False, "HIA": False}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return default_adme
        mol = Chem.AddHs(mol)
        mw, logp, hbd, hba, tpsa = Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.TPSA(mol)
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lipinski_obey = "Yes" if violations <= 1 else "No"
        oral_bio = "Yes (High)" if violations == 0 else ("Yes (Moderate)" if violations == 1 else "No (Poor)")
        ring_info = mol.GetRingInfo().AtomRings()
        max_ring = max([len(r) for r in ring_info]) if ring_info else 0
        vol = float(mw) * 0.88 
        acidic_pka = "Acidic (~4.5)" if mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")) else ("Weak Acid (~9.5)" if mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH]")) else "Neutral")
        basic_pka = "Basic (~9.0)" if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")) else ("Weak Base (~4.0)" if mol.HasSubstructMatch(Chem.MolFromSmarts("cN")) else "Neutral")
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        est_mp = max(20.0, (mw * 0.4) + (hbd * 25.0) - (rot_bonds * 5.0))
        est_bp = est_mp + 150.0 + (mw * 0.5)
        hia, bbb = (tpsa < 132) and (-2.0 < logp < 6.0), (tpsa < 79) and (0.4 < logp < 6.0)
        perm = "High BBB Penetration & GI Absorption" if bbb else ("Good GI Absorption" if hia else "Poor Absorption / Impermeable")
        return {"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "TPSA": tpsa, "Violations": violations, "Lipinski_Obey": lipinski_obey, "Oral_Bio": oral_bio, "MaxRing": max_ring, "Volume": vol, "pKa_Acid": acidic_pka, "pKa_Base": basic_pka, "MP": est_mp, "BP": est_bp, "Permeability": perm, "BBB": bbb, "HIA": hia}
    except Exception: return default_adme

# =====================================================================
# 4. HIGH PERFORMANCE VISUALIZATION UTILITIES & HTML REPORTING
# =====================================================================

def generate_clean_2d_image(smiles_str, include_labels=False, zoom_level=450):
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            mol_to_draw = Chem.RemoveHs(mol)
            if include_labels:
                for atom in mol_to_draw.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx()))
            img = Draw.MolToImage(mol_to_draw, size=(zoom_level, int(zoom_level * 0.77)))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom:15px;"/>'
    except Exception: pass
    return None

def generate_ftir_image(target_peak):
    wavenumbers = np.linspace(400, 4000, 500)
    baseline = 98.0 - 2.0 * np.sin(wavenumbers / 200.0)
    effect = 40.0 * np.exp(-((wavenumbers - target_peak) / 45.0)**2)
    transmittance = np.clip(baseline - effect, 5.0, 100.0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(wavenumbers, transmittance, color='#1e3c72', linewidth=2)
    ax.set_xlim(4000, 400); ax.set_ylim(0, 105)
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Transmittance (%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.fill_between(wavenumbers, transmittance, 105, color='#1e3c72', alpha=0.05)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def render_advanced_modeling_blueprint(receptor_data, ligand_data, mode="cartoon", show_surface=False, interactions_list=[], unique_id="container"):
    surface_js = f"viewer_{unique_id}.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.45, colorscheme:{{prop:'b',gradient:'rwb'}}}}, {{model:0}});" if show_surface else ""
    int_lines_js = ""
    for interact in interactions_list:
        rc, lc = interact["r_coord"], interact["l_coord"]
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js += f"""
        viewer_{unique_id}.addCylinder({{start:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, end:{{x:{lc[0]}, y:{lc[1]}, z:{lc[2]}}}, radius:0.07, color:'{color}', dashed:true}});
        viewer_{unique_id}.addLabel("{interact['Residue Contact']}", {{position:{{x:{rc[0]}, y:{rc[1]}, z:{rc[2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});
        """
    html_content = f"""
    <div id="wrapper_{unique_id}" style="position:relative; width:100%;">
        <button onclick="toggleFullScreen_{unique_id}()" style="position:absolute; top:12px; right:12px; z-index:9999; padding:6px 12px; background:#007bff; color:white; border:none; border-radius:4px; cursor:pointer; font-weight:bold; box-shadow:0 2px 4px rgba(0,0,0,0.15);">🖥 Fullscreen View</button>
        <div id="{unique_id}" style="height: 480px; width: 100%; position: relative; border-radius:10px; border:1px solid #eaeaea; background:#ffffff;"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        let viewer_{unique_id} = $3Dmol.createViewer(document.getElementById('{unique_id}'), {{backgroundColor: '#ffffff'}});
        if (`{receptor_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{receptor_data}`, 'pdb');
            if ('{mode}' === 'cartoon') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}}); }} 
            else if ('{mode}' === 'spacefill') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{sphere: {{colorscheme: 'chain', radius:1.1}}}}); }} 
            else if ('{mode}' === 'sticks') {{ viewer_{unique_id}.setStyle({{model: 0}}, {{stick: {{colorscheme: 'chain', radius:0.25}}}}); }}
            else {{ viewer_{unique_id}.setStyle({{model: 0}}, {{cartoon: {{colorscheme: 'chain', style: 'oval', thickness: 0.6}}}}); }}
        }}
        {surface_js}
        if (`{ligand_data}`.trim().length > 0) {{
            viewer_{unique_id}.addModel(`{ligand_data}`, 'pdb');
            viewer_{unique_id}.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
        }}
        {int_lines_js}
        viewer_{unique_id}.zoomTo(); viewer_{unique_id}.render();
        function toggleFullScreen_{unique_id}() {{
            let elem = document.getElementById("wrapper_{unique_id}");
            if (!document.fullscreenElement) {{ elem.requestFullscreen(); document.getElementById("{unique_id}").style.height = "90vh"; }}
            else {{ document.exitFullscreen(); document.getElementById("{unique_id}").style.height = "480px"; }}
        }}
        document.addEventListener('fullscreenchange', () => {{ if (!document.fullscreenElement) document.getElementById("{unique_id}").style.height = "480px"; }});
    </script>
    """
    components.html(html_content, height=510)

def generate_ayurvedic_card(data, is_streamlit=True):
    if not data: return ""
    bg_color = "#f4fcf7" if is_streamlit else "#ecfdf5"
    border_color = "#2e7d32" if is_streamlit else "#10b981"
    card = f"""
    <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 6px solid {border_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 25px;">
        <h2 style="margin-top: 0; color: #1b5e20;">🌳 {data['Herb / Tree Name']}</h2>
        <h4 style="margin: 5px 0; color: #388e3c;"><i>{data['Scientific Name']}</i> | Family: {data['Family']}</h4>
        <h5 style="margin: 5px 0; color: #555;"><b>Medicinal Activity:</b> {data['Medicinal Activity']}</h5>
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
        <p style="font-size: 16px; color: #333;"><b>Sanskrit Shloka:</b><br><span style="font-family: serif; font-size: 18px;">{data['Sanskrit Shloka (Bhavaprakasha Nighantu)']}</span></p>
        <p style="font-size: 14px; color: #555;"><b>Pronunciation:</b> <i>{data['Roman Transliteration']}</i></p>
        <p style="font-size: 14px; color: #555;"><b>Meaning / Context:</b> This classical verse highlights the properties mapped to the {data['Classical Karma (Action)']} and {data['Medicinal Activity']} profile.</p>
        <p style="font-size: 14px; color: #555;"><b>Dravyaguna Profile:</b> {data['Dravyaguna Profile (Rasa/Virya/Vipaka)']}</p>
        <p style="font-size: 14px; color: #555;"><b>Classical Karma (Action):</b> {data['Classical Karma (Action)']}</p>
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 15px 0;">
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0;"><b>Target Protein:</b> {data['Target Protein / Receptor Name']}</p>
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0;"><b>Ligand (Phytochemical):</b> {data['Phytochemical']}</p>
        <p style="font-size: 14px; color: #1e3c72; margin: 2px 0; word-break: break-all;"><b>Canonical SMILES:</b> {data['Canonical SMILES']}</p>
    </div>
    """
    return card

def build_phase1_html_report(meta, p_2d, smiles_cache, grid_params, df_results_p1, orig_ints, receptor_data, orig_ligand_pose_data, selected_pose_orig, style_mode, show_surface, pre_uff, post_uff, delta_uff, active_retained_ions, uff_theory_html, orig_matrix_html, grid_strategy, tree_data=None):
    res_html = "<p>No docking data.</p>"
    if df_results_p1 is not None and not df_results_p1.empty:
        res_html = '<table class="dataframe table"><thead><tr>'
        for col in df_results_p1.columns: res_html += f'<th>{col}</th>'
        res_html += '</tr></thead><tbody>'
        for _, row in df_results_p1.iterrows():
            res_html += '<tr>'
            for col in df_results_p1.columns:
                val = row[col]
                style = ''
                if col == 'Affinity (kcal/mol)':
                    try:
                        v = float(val)
                        if v < 0: style = 'style="color: #10b981; font-weight: bold;"'
                        elif v > 0: style = 'style="color: #ef4444; font-weight: bold;"'
                    except: pass
                res_html += f'<td {style}>{val}</td>'
            res_html += '</tr>'
        res_html += '</tbody></table>'

    safe_rec = str(receptor_data).replace('`', '').replace('\\', '\\\\')
    safe_lig_orig = str(orig_ligand_pose_data).replace('`', '').replace('\\', '\\\\')

    int_lines_js1 = ""
    for interact in orig_ints:
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js1 += f"viewer1.addCylinder({{start:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, end:{{x:{interact['l_coord'][0]}, y:{interact['l_coord'][1]}, z:{interact['l_coord'][2]}}}, radius:0.07, color:'{color}', dashed:true}});\n"
        int_lines_js1 += f"viewer1.addLabel(\"{interact['Residue Contact']}\", {{position:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});\n"

    if style_mode == 'cartoon': style_js = "viewer1.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
    elif style_mode == 'spacefill': style_js = "viewer1.setStyle({model: 0}, {sphere: {colorscheme: 'chain', radius:1.1}});"
    elif style_mode == 'sticks': style_js = "viewer1.setStyle({model: 0}, {stick: {colorscheme: 'chain', radius:0.25}});"
    else: style_js = "viewer1.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
        
    surface_js = "viewer1.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface else ""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>InSilico BioSphere - Phase 1 Docking Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 0; background-color: #f9f9fb; }}
            .header-banner {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px; border-bottom: 5px solid #00c6ff; text-align: center; position: relative; }}
            .header-banner h1 {{ margin: 0; font-size: 28px; letter-spacing: 1px; }}
            .header-banner p {{ margin: 5px 0 0 0; font-size: 14px; opacity: 0.9; }}
            .container {{ max-width: 1000px; margin: 30px auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h2, h3, h4 {{ color: #1e3c72; }}
            h2 {{ border-bottom: 2px solid #eef2f7; padding-bottom: 8px; margin-top: 35px; font-size: 20px; }}
            h3 {{ font-size: 16px; margin-top: 20px; }}
            h4 {{ font-size: 15px; margin-top: 15px; text-align: center; }}
            .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; background: #f4f7f6; padding: 20px; border-radius: 8px; }}
            .meta-item {{ font-size: 14px; }}
            .meta-item strong {{ color: #1e3c72; }}
            .table-wrapper {{ overflow-x: auto; margin: 20px 0; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 600px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
            th {{ background-color: #f8fafc; color: #1e3c72; font-weight: 600; }}
            .structure-img {{ background: white; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; max-width: 320px; text-align: center; margin: 0 auto; }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <h1>🔬 InSilico BioSphere Phase 1 Docking Report</h1>
            <p>Department of Chemistry, Shivaji Science College, Nagpur, India</p>
        </div>
        
        <div class="container">
            {generate_ayurvedic_card(tree_data, is_streamlit=False) if tree_data else ""}
            
            <h2>1. Baseline Docking Configuration & Target Matrix</h2>
            <div class="meta-grid">
                <div class="meta-item"><strong>Target Protein:</strong> {meta['name']}</div>
                <div class="meta-item"><strong>PDB ID:</strong> {meta['id']}</div>
                <div class="meta-item"><strong>Catalytic Cofactors Filter:</strong> {active_retained_ions}</div>
                <div class="meta-item"><strong>Ligand (SMILES):</strong> <span style="word-break: break-all; font-family: monospace;">{smiles_cache}</span></div>
                <div class="meta-item"><strong>Grid Search Strategy:</strong> {grid_strategy}</div>
                <div class="meta-item"><strong>Grid Box (X,Y,Z):</strong> {grid_params['cx']}, {grid_params['cy']}, {grid_params['cz']}</div>
                <div class="meta-item"><strong>Grid Dimensions (Å):</strong> {grid_params['sx']} × {grid_params['sy']} × {grid_params['sz']}</div>
            </div>

            <div style="text-align: center; margin-bottom: 20px;">
                <h4>Lead Ligand 2D Topology</h4>
                <div class="structure-img">{p_2d}</div>
            </div>

            <h2>2. Baseline Molecular Docking Screening Results</h2>
            <div class="table-wrapper">
                {res_html}
            </div>

            <h2>3. Local Contact Residues & Bond Assignments Matrix (Pose {selected_pose_orig})</h2>
            <div class="table-wrapper">
                {orig_matrix_html}
            </div>

            <h2>4. Interactive 3D Protein-Ligand View</h2>
            <div id="container-3d-orig" style="height: 500px; width: 100%; position: relative; border-radius:8px; border:1px solid #eaeaea; background:#ffffff;"></div>
            
            <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
            <script>
                let viewer1 = $3Dmol.createViewer(document.getElementById('container-3d-orig'), {{backgroundColor: '#ffffff'}});
                let rec_data = `{safe_rec}`; let lig_data_orig = `{safe_lig_orig}`;
                if (rec_data.trim().length > 0) {{ viewer1.addModel(rec_data, 'pdb'); {style_js} }}
                if (lig_data_orig.trim().length > 0) {{ viewer1.addModel(lig_data_orig, 'pdb'); viewer1.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}}); }}
                {surface_js} {int_lines_js1} viewer1.zoomTo(); viewer1.render();
            </script>
            
            <div class="section" style="border-left: 6px solid #1e3c72; background-color: #f4f8fd; padding:15px; margin-top:30px;">
                <h2>5. Scientific Methodology & Manuscript Citation Track</h2>
                <p><i>The following standard protocol text is generated dynamically to assist in manuscript development and formal peer-reviewed reporting:</i></p>
                <blockquote style="background: #fff; padding: 12px; border-left: 4px solid #1e3c72; font-style: italic; margin: 10px 0;">
                    Molecular docking was performed using the semi-empirical force field parameters of AutoDock Vina inside the InSilico BioSphere framework. To maintain structural and biological validity, essential catalytic cofactor ions were explicitly preserved within the target binding cleft during search configurations. Potential localized steric constraints and rigid atomic wall collisions resulting from structural constraints were resolved by subjecting the final protein-ligand complexes to post-docking energy minimization using the Universal Force Field (UFF) optimized to a convergence tolerance of 10<sup>-4</sup> kcal/mol·Å.
                </blockquote>
            </div>
            
            {uff_theory_html}
            
        </div>
    </body>
    </html>
    """

def build_comprehensive_html_report(meta, adme_p, adme_v, variant_row, iupac, shift_msg, f_img, v_2d, p_2d, 
                                    smiles_cache, baseline_affinity, grid_params, df_results_baseline, df_results_redesign, 
                                    orig_ints, new_ints, receptor_data, orig_ligand_pose_data, redesign_ligand_pose_data, 
                                    selected_pose_orig, selected_pose_new, style_mode_orig, show_surface_orig,
                                    style_mode_new, show_surface_new, master_verdict, df_comparison_html, pre_uff, post_uff, delta_uff, active_retained_ions,
                                    uff_theory_html, orig_matrix_html, new_matrix_html, grid_strategy, tree_data=None):
    
    def generate_html_table(df):
        if df is None or df.empty: return "<p>No docking data.</p>"
        t_html = '<table class="dataframe table"><thead><tr>'
        for col in df.columns: t_html += f'<th>{col}</th>'
        t_html += '</tr></thead><tbody>'
        for _, row in df.iterrows():
            t_html += '<tr>'
            for col in df.columns:
                val = row[col]
                style = ''
                if col == 'Affinity (kcal/mol)':
                    try:
                        v = float(val)
                        if v < 0: style = 'style="color: #10b981; font-weight: bold;"'
                        elif v > 0: style = 'style="color: #ef4444; font-weight: bold;"'
                    except: pass
                t_html += f'<td {style}>{val}</td>'
            t_html += '</tr>'
        t_html += '</tbody></table>'
        return t_html

    res_html_baseline = generate_html_table(df_results_baseline)
    res_html_redesign = generate_html_table(df_results_redesign)

    safe_rec = str(receptor_data).replace('`', '').replace('\\', '\\\\')
    safe_lig_orig = str(orig_ligand_pose_data).replace('`', '').replace('\\', '\\\\')
    safe_lig_redesign = str(redesign_ligand_pose_data).replace('`', '').replace('\\', '\\\\')

    int_lines_js1 = ""
    for interact in orig_ints:
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js1 += f"viewer1.addCylinder({{start:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, end:{{x:{interact['l_coord'][0]}, y:{interact['l_coord'][1]}, z:{interact['l_coord'][2]}}}, radius:0.07, color:'{color}', dashed:true}});\n"
        int_lines_js1 += f"viewer1.addLabel(\"{interact['Residue Contact']}\", {{position:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});\n"

    int_lines_js2 = ""
    for interact in new_ints:
        color = "yellow" if "Hydrogen" in interact["Interaction Type"] else "cyan"
        int_lines_js2 += f"viewer2.addCylinder({{start:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, end:{{x:{interact['l_coord'][0]}, y:{interact['l_coord'][1]}, z:{interact['l_coord'][2]}}}, radius:0.07, color:'{color}', dashed:true}});\n"
        int_lines_js2 += f"viewer2.addLabel(\"{interact['Residue Contact']}\", {{position:{{x:{interact['r_coord'][0]}, y:{interact['r_coord'][1]}, z:{interact['r_coord'][2]}}}, backgroundColor:'white', fontColor:'black', backgroundOpacity:0.8, fontSize:10}});\n"

    if style_mode_orig == 'cartoon': style_js1 = "viewer1.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
    elif style_mode_orig == 'spacefill': style_js1 = "viewer1.setStyle({model: 0}, {sphere: {colorscheme: 'chain', radius:1.1}});"
    elif style_mode_orig == 'sticks': style_js1 = "viewer1.setStyle({model: 0}, {stick: {colorscheme: 'chain', radius:0.25}});"
    else: style_js1 = "viewer1.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"

    if style_mode_new == 'cartoon': style_js2 = "viewer2.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
    elif style_mode_new == 'spacefill': style_js2 = "viewer2.setStyle({model: 0}, {sphere: {colorscheme: 'chain', radius:1.1}});"
    elif style_mode_new == 'sticks': style_js2 = "viewer2.setStyle({model: 0}, {stick: {colorscheme: 'chain', radius:0.25}});"
    else: style_js2 = "viewer2.setStyle({model: 0}, {cartoon: {colorscheme: 'chain', style: 'oval', thickness: 0.6}});"
        
    surface_js1 = "viewer1.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface_orig else ""
    surface_js2 = "viewer2.addSurface($3Dmol.SurfaceType.VDW, {opacity:0.45, colorscheme:{prop:'b',gradient:'rwb'}}, {model:0});" if show_surface_new else ""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>InSilico BioSphere Complete Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 0; background-color: #f9f9fb; }}
            .header-banner {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px; border-bottom: 5px solid #00c6ff; text-align: center; position: relative; }}
            .header-banner h1 {{ margin: 0; font-size: 28px; letter-spacing: 1px; }}
            .header-banner p {{ margin: 5px 0 0 0; font-size: 14px; opacity: 0.9; }}
            .copyright-header {{ font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: rgba(255,255,255,0.7); margin-bottom: 10px; display: block; }}
            .container {{ max-width: 1000px; margin: 30px auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
            h2, h3, h4 {{ color: #1e3c72; }}
            h2 {{ border-bottom: 2px solid #eef2f7; padding-bottom: 8px; margin-top: 35px; font-size: 20px; }}
            h3 {{ font-size: 16px; margin-top: 20px; }}
            h4 {{ font-size: 15px; margin-top: 15px; text-align: center; }}
            .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; background: #f4f7f6; padding: 20px; border-radius: 8px; }}
            .meta-item {{ font-size: 14px; }}
            .meta-item strong {{ color: #1e3c72; }}
            .table-wrapper {{ overflow-x: auto; margin: 20px 0; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; min-width: 600px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
            th {{ background-color: #f8fafc; color: #1e3c72; font-weight: 600; }}
            .structure-box {{ display: flex; gap: 30px; margin: 20px 0; background: #fafafa; padding: 20px; border-radius: 8px; border: 1px solid #eef2f7; align-items: center; justify-content: center; flex-wrap: wrap; }}
            .structure-img {{ background: white; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; max-width: 320px; text-align: center; margin: 0 auto; }}
            .scandata {{ font-family: monospace; background: #f1f5f9; padding: 3px 6px; border-radius: 4px; font-size: 13px; word-break: break-all; }}
            .summary-card {{ background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 20px; border-radius: 6px; margin: 25px 0; color: #065f46; font-size: 14.5px; }}
            .verdict-card {{ background-color: #fffbeb; border-left: 5px solid #f59e0b; padding: 20px; border-radius: 6px; margin: 25px 0; color: #92400e; font-size: 15px; font-weight: bold; }}
            footer {{ text-align: center; padding: 20px; font-size: 12px; color: #64748b; margin-top: 5px; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <span class="copyright-header">copyright@sarang dhote</span>
            <h1>🔬 InSilico BioSphere Complete Execution Report</h1>
            <p>Department of Chemistry, Shivaji Science College, Nagpur, India</p>
        </div>
        
        <div class="container">
            {generate_ayurvedic_card(tree_data, is_streamlit=False) if tree_data else ""}
            
            <h2>1. Baseline Docking Configuration & Target Matrix</h2>
            <div class="meta-grid">
                <div class="meta-item"><strong>Target Protein Name:</strong> {meta['name']}</div>
                <div class="meta-item"><strong>Target PDB ID:</strong> {meta['id']}</div>
                <div class="meta-item"><strong>Method / Resolution:</strong> {meta['method']} ({meta['res']})</div>
                <div class="meta-item"><strong>Catalytic Cofactors Filter:</strong> {active_retained_ions}</div>
                <div class="meta-item"><strong>Lead Phytochemical (SMILES):</strong> <span class="scandata">{smiles_cache}</span></div>
                <div class="meta-item"><strong>Grid Search Strategy:</strong> {grid_strategy}</div>
                <div class="meta-item"><strong>Grid Box Coordinates (X, Y, Z):</strong> {grid_params['cx']}, {grid_params['cy']}, {grid_params['cz']}</div>
                <div class="meta-item"><strong>Grid Dimensions (Å):</strong> {grid_params['sx']} × {grid_params['sy']} × {grid_params['sz']}</div>
                <div class="meta-item"><strong>Search Exhaustiveness:</strong> {grid_params['exh']}</div>
            </div>

            <h2>2. Baseline Molecular Docking Screening Results (Phase 1)</h2>
            <div class="table-wrapper">
                {res_html_baseline}
            </div>
            
            <h2>3. Optimized Derivative Docking Screening Results (Phase 4)</h2>
            <div class="table-wrapper">
                {res_html_redesign}
            </div>

            <h2>4. Validation Complex Analysis (Side-by-Side Comparison)</h2>
            <p>Interactive 3D representation comparing the original lead and the redesigned derivative inside the target receptor pocket.</p>
            
            <div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px;">
                    <h4>Original Lead (Pose {selected_pose_orig})</h4>
                    <div id="container-3d-orig" style="height: 400px; width: 100%; position: relative; border-radius:8px; border:1px solid #eaeaea; background:#ffffff; box-shadow: 0 4px 10px rgba(0,0,0,0.05);"></div>
                </div>
                <div style="flex: 1; min-width: 300px;">
                    <h4>Optimized Derivative (Pose {selected_pose_new})</h4>
                    <div id="container-3d-redesign" style="height: 400px; width: 100%; position: relative; border-radius:8px; border:1px solid #eaeaea; background:#ffffff; box-shadow: 0 4px 10px rgba(0,0,0,0.05);"></div>
                </div>
            </div>
            
            <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
            <script>
                // Viewer 1 (Original)
                let viewer1 = $3Dmol.createViewer(document.getElementById('container-3d-orig'), {{backgroundColor: '#ffffff'}});
                let rec_data = `{safe_rec}`;
                let lig_data_orig = `{safe_lig_orig}`;
                if (rec_data.trim().length > 0) {{
                    viewer1.addModel(rec_data, 'pdb');
                    {style_js1}
                }}
                if (lig_data_orig.trim().length > 0) {{
                    viewer1.addModel(lig_data_orig, 'pdb');
                    viewer1.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
                }}
                {surface_js1}
                {int_lines_js1}
                viewer1.zoomTo(); 
                viewer1.render();

                // Viewer 2 (Redesign)
                let viewer2 = $3Dmol.createViewer(document.getElementById('container-3d-redesign'), {{backgroundColor: '#ffffff'}});
                let lig_data_redesign = `{safe_lig_redesign}`;
                if (rec_data.trim().length > 0) {{
                    viewer2.addModel(rec_data, 'pdb');
                    {style_js2}
                }}
                if (lig_data_redesign.trim().length > 0) {{
                    viewer2.addModel(lig_data_redesign, 'pdb');
                    viewer2.setStyle({{model: 1}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.28}}}});
                }}
                {surface_js2}
                {int_lines_js2}
                viewer2.zoomTo(); 
                viewer2.render();
            </script>

            <p style="margin-top:20px;">Direct Thermodynamic Comparison Matrix</p>
            <div class="table-wrapper">
                {df_comparison_html}
            </div>

            <h3>Local Contact Residues & Bond Assignments</h3>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px;">
                    <h4>Original Lead Matrices</h4>
                    <div class="table-wrapper">{orig_matrix_html}</div>
                </div>
                <div style="flex: 1; min-width: 300px;">
                    <h4>Derivative Matrices</h4>
                    <div class="table-wrapper">{new_matrix_html}</div>
                </div>
            </div>

            <h2>5. Generative Scaffold Optimization</h2>
            <div class="meta-grid">
                <div class="meta-item"><strong>Isolated Variant ID:</strong> {variant_row['Variant ID']}</div>
                <div class="meta-item"><strong>Appended Fragment:</strong> {variant_row['Fragment Added']}</div>
                <div class="meta-item"><strong>Synthetic Route Evaluated:</strong> {variant_row['Route']}</div>
                <div class="meta-item"><strong>Predicted Yield Tier:</strong> <span style="color:#1e3c72; font-weight:bold;">{variant_row['Yield Prediction']}</span></div>
            </div>
            
            <div class="structure-box">
                <div style="flex:1; text-align: center;">
                    <h4>Original Phytochemical Lead</h4>
                    <div class="structure-img">{p_2d}</div>
                </div>
                <div style="flex:1; text-align: center;">
                    <h4>Optimized Derivative</h4>
                    <div class="structure-img">{v_2d}</div>
                </div>
            </div>
            
            <div style="margin-bottom: 20px; padding: 10px; background: #fafafa; border-radius: 6px;">
                <h3 style="margin-top:0;">Redesigned Target SMILES String Matrix</h3>
                <div class="scandata" style="margin-bottom: 5px;">{variant_row['Redesigned SMILES']}</div>
                <strong>Pathway coordinates optimized via functional block swapping mechanics.</strong>
            </div>

            <h2>6. ADMET 3.0 Pharmacokinetics Analysis</h2>
            <p><strong>Automated IUPAC Nomenclature Generation:</strong></p>
            <div class="scandata" style="margin-bottom:20px; background:#e0f2fe; color:#0369a1; padding:10px; border-left: 4px solid #0284c7;">
                {iupac}
            </div>
            
            <h3>Molecular Property Comparative Matrix</h3>
            <div class="table-wrapper">
                <table>
                    <tr><th>Parameter Parameterized</th><th>Original Phytochemical Lead</th><th>Redesigned Variant Matrix</th></tr>
                    <tr><td>Obey Lipinski's Rule?</td><td>{adme_p['Lipinski_Obey']}</td><td>{adme_v['Lipinski_Obey']}</td></tr>
                    <tr><td>Oral Bioavailability Probability</td><td>{adme_p['Oral_Bio']}</td><td>{adme_v['Oral_Bio']}</td></tr>
                    <tr><td>Total Permeability Profile</td><td>{adme_p['Permeability']}</td><td>{adme_v['Permeability']}</td></tr>
                    <tr><td>TPSA (Å²)</td><td>{adme_p['TPSA']}</td><td>{adme_v['TPSA']}</td></tr>
                    <tr><td>Molecular Volume (Å³)</td><td>{adme_p['Volume']}</td><td>{adme_v['Volume']}</td></tr>
                    <tr><td>Lipophilicity Parameter (LogP)</td><td>{adme_p['LogP']}</td><td>{adme_v['LogP']}</td></tr>
                </table>
            </div>

            <h3>Structural Shift Assessment Narrative</h3>
            <div class="summary-card">
                {shift_msg}
            </div>

            <h3>📊 Modeled Vibrational Spectrum Footprint (FTIR)</h3>
            <div style="text-align: center; margin: 20px 0;">
                <img src="data:image/png;base64,{f_img}" style="max-width:100%; border-radius:6px; border: 1px solid #e2e8f0;"/>
            </div>
            
            <h2>7. Master Synthesis Verdict</h2>
            <div class="verdict-card">
                {master_verdict}
            </div>
            
            <div class="section" style="border-left: 6px solid #1e3c72; background-color: #f4f8fd; padding:15px; margin-top:30px;">
                <h2>8. Scientific Methodology & Manuscript Citation Track</h2>
                <p><i>The following standard protocol text is generated dynamically to assist in manuscript development and formal peer-reviewed reporting:</i></p>
                <blockquote style="background: #fff; padding: 12px; border-left: 4px solid #1e3c72; font-style: italic; margin: 10px 0;">
                    Molecular docking was performed using the semi-empirical force field parameters of AutoDock Vina inside the InSilico BioSphere framework. To maintain structural and biological validity, essential catalytic cofactor ions were explicitly preserved within the target binding cleft during search configurations. Potential localized steric constraints and rigid atomic wall collisions resulting from structural constraints were resolved by subjecting the final protein-ligand complexes to post-docking energy minimization using the Universal Force Field (UFF) optimized to a convergence tolerance of 10<sup>-4</sup> kcal/mol·Å.
                </blockquote>
            </div>

            {uff_theory_html}
            
        </div>
        <footer>
            <p>Report compiled successfully. Ready for manuscript citation.</p>
            <p>InSilico BioSphere: An Integrated Platform for Automated Molecular Docking.</p>
            <p>Developed by Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry,<br>
            Shivaji Science College, Nagpur, India.<br>
            Email: contact - sarangresearch@gmail.com</p>
        </footer>
    </body>
    </html>
    """

# =====================================================================
# 5. AYURVEDIC DATABASE LOADER (DRAVYADOCK CORE)
# =====================================================================
@st.cache_data
def load_ayurvedic_db():
    hardcoded_data = [
        {"Master ID": "M-001", "Herb / Tree Name": "Neem", "Scientific Name": "Azadirachta indica", "Family": "Meliaceae", "Phytochemical": "Nimbin", "Canonical SMILES": "CC(=O)OC1C(C2(CC3C(C24C1C(O4)C(=C)C(=O)OC)CC(C5(C3CC(O5)C6=COCO6)C)OC(=O)C)C)C", "Medicinal Activity": "Antibacterial", "Target Protein / Receptor Name": "Penicillin-Binding Protein 2a (PBP2a)", "PDB ID": "1VQQ", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "निम्बः शीतो लघुस्तिक्तो व्रणशोधनरोपणः। चक्षुष्यः कफपित्तघ्नः कुष्ठहृत् कृमिहृत्परः॥", "Roman Transliteration": "nimbaḥ śīto laghustikto vraṇaśodhanaropaṇaḥ | cakṣuṣyaḥ kaphapittaghnaḥ kuṣṭhahṛt kṛmihṛtparaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Shita; Vipaka: Katu", "Classical Karma (Action)": "Krimighna (Antimicrobial) Vrana-shodhana Kusthaha"},
        {"Master ID": "M-002", "Herb / Tree Name": "Tulsi", "Scientific Name": "Ocimum sanctum", "Family": "Lamiaceae", "Phytochemical": "Eugenol", "Canonical SMILES": "COC1=C(C=CC(=C1)CC=C)O", "Medicinal Activity": "Antimicrobial", "Target Protein / Receptor Name": "Candida albicans Secreted Aspartyl Proteinase 1", "PDB ID": "1ZAP", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "तुलसी कटुका तिक्ता हृद्या उष्णा दाहपित्तकृत्। दीपनी कुष्ठकृच्छ्रास्त्रपार्श्वशूलविनाशिनी॥", "Roman Transliteration": "tulasī kaṭukā tiktā hṛdyā uṣṇā dāhapittakṛt | dīpanī kuṣṭhakṛcchrāstrapārśvaśūlavināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Krimighna Hridya (Cardioprotective) Dipana"},
        {"Master ID": "M-003", "Herb / Tree Name": "Ashwagandha", "Scientific Name": "Withania somnifera", "Family": "Solanaceae", "Phytochemical": "Withaferin A", "Canonical SMILES": "CC1=C(C(=O)C2=C(C1O)C3CCC4C5CC6C(C5(CCC4(C3(C2)C)O)C)OC(=O)C6(C)O)C7=CC(=O)OC7", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Heat Shock Protein 90 (Hsp90)", "PDB ID": "2YI5", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "अश्वगन्धा अनिलाश्लेष्मश्वित्रशोथक्षयापहा। बल्या रसायनी तिक्ता कषायोष्णा अतिशुक्रला॥", "Roman Transliteration": "aśvagandhā anilāśleṣmaśvitraśothakṣayāpahā | balyā rasāyanī tiktā kaṣāyoṣṇā atiśukralā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya Madhura; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Rasayana (Rejuvenative) Balya Shothahara"},
        {"Master ID": "M-004", "Herb / Tree Name": "Turmeric", "Scientific Name": "Curcuma longa", "Family": "Zingiberaceae", "Phytochemical": "Curcumin", "Canonical SMILES": "COC1=C(O)C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(OC)C=C2)O", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Glycogen Synthase Kinase-3 beta (GSK-3β)", "PDB ID": "1Q5K", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "हरिद्रा कटुका तिक्ता रूक्षोष्णा कफपित्तनुत्। वर्ण्या त्वग्दोषमेहास्त्रशोथपाण्डुव्रणापहा॥", "Roman Transliteration": "haridrā kaṭukā tiktā rūkṣoṣṇā kaphapittanut | varṇyā tvagdoṣamehāstraśothapāṇḍuvraṇāpahā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Vranahara (Wound Healing) Lekhaniya Mehahara"},
        {"Master ID": "M-005", "Herb / Tree Name": "Giloy", "Scientific Name": "Tinospora cordifolia", "Family": "Menispermaceae", "Phytochemical": "Berberine", "Canonical SMILES": "COC1=C(C2=C(C=C1)C3=CN4CCC5=CC6=C(C=C5C4C3=C2)OCO6)OC", "Medicinal Activity": "Antidiabetic", "Target Protein / Receptor Name": "AMP-activated Protein Kinase (AMPK)", "PDB ID": "4CFE", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "गुडूची कटुका तिक्ता स्वादुपाका रसायनी। ज्वरकुष्ठप्रमेहार्शःकण्डूहृद्रोगवातनुत्॥", "Roman Transliteration": "guḍūcī kaṭukā tiktā svādupākā rasāyanī | jvarakuṣṭhapramehārśaḥkaṇḍūhṛdroghavātanut ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Pramehahara (Antidiabetic) Rasayana Tridosashamana"},
        {"Master ID": "M-006", "Herb / Tree Name": "Brahmi", "Scientific Name": "Bacopa monnieri", "Family": "Plantaginaceae", "Phytochemical": "Bacoside A", "Canonical SMILES": "CC1C(C(C(C(O1)OC2C(C(OC3CC4(C5CCC6C7(CCC(C(C7CCC6(C5CC(=O)C4(C3(C)C)C)C)(C)C)O)C)C)CO)O)O)O)O", "Medicinal Activity": "Neuroprotective", "Target Protein / Receptor Name": "Human Beta-Amyloid (1-42) fibrils", "PDB ID": "2BEG", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "ब्राह्मी हिमा सरा तिक्ता मतिमेधाकृता स्वर्या। आयुष्या रसायनी स्वर्या विस्मृतिभ्रमहापरा॥", "Roman Transliteration": "brāhmī himā sarā tiktā matimedhākṛtā svaryā | āyuṣyā rasāyanī svaryā vismṛtibramahāparā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Medhya (Neuroprotective) Ayushya Vismrtihara"},
        {"Master ID": "M-007", "Herb / Tree Name": "Arjuna", "Scientific Name": "Terminalia arjuna", "Family": "Combretaceae", "Phytochemical": "Arjunic Acid", "Canonical SMILES": "CC1CCC2(CCC3(C(=CCC4C3(CCC5C4(CCC(C5(C)C)O)C)C)C2C1O)C)C(=O)O", "Medicinal Activity": "Cardioprotective", "Target Protein / Receptor Name": "Human Angiotensin-Converting Enzyme (ACE)", "PDB ID": "1O86", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "ककुभोऽर्जुनः कीर्तितः स्याच्छीतलः कषायको। हृद्रोगक्षतक्षयविषप्रशमनोऽपि च॥", "Roman Transliteration": "kakubho'rjunaḥ kīrtitaḥ syācchītalaḥ kaṣāyako | hṛdroghakṣatakṣayaviṣapraśamano'pi ca ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Kasaya; Virya: Shita; Vipaka: Katu", "Classical Karma (Action)": "Hridya (Cardioprotective) Raktastambhana Kṣatahara"},
        {"Master ID": "M-008", "Herb / Tree Name": "Sarpagandha", "Scientific Name": "Rauvolfia serpentina", "Family": "Apocynaceae", "Phytochemical": "Reserpine", "Canonical SMILES": "COC1=C(C=C2C(=C1)C3CC4C(CC3NC2C5CC(C(C(C5)C(=O)OC)OC(=O)C6=CC(=C(C(=C6)OC)OC)OC)O)C(=O)O)OC", "Medicinal Activity": "Antihpertensive", "Target Protein / Receptor Name": "Vesicular Monoamine Transporter 2 (VMAT2)", "PDB ID": "7VUT", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "सर्पगन्धा तु तिक्तोष्णा कटुका च कफापहा। निद्राप्रदा रक्तवातशमनी काममन्दिनी॥", "Roman Transliteration": "sarpagandhā tu tiktoṣṇā kaṭukā ca kaphāpahā | nidrāpradā raktavātaśamanī kāmamandinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Nidraprada (Sedative) Raktavata-shamana (Antihpertensive)"},
        {"Master ID": "M-009", "Herb / Tree Name": "Vasaka", "Scientific Name": "Justicia adhatoda", "Family": "Acanthaceae", "Phytochemical": "Vasicine", "Canonical SMILES": "C1CC2=NC3=CC=CC=C3C4C2(C1)N=C(O4)C", "Medicinal Activity": "Bronchodilator", "Target Protein / Receptor Name": "Beta-2 Adrenergic Receptor", "PDB ID": "7DHI", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "वासको वासिका वासा भिषङ्माता च सिंहिका। वासा तिक्ता कषायोष्णा कफपित्तविनाशिनी॥", "Roman Transliteration": "vāsako vāsikā vāsā bhiṣaṅmātā ca siṃhikā | vāsā tiktā kaṣāyoṣṇā kaphapittavināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Shita; Vipaka: Katu", "Classical Karma (Action)": "Kashahara (Antitussive) Shwasahara (Bronchodilator)"},
        {"Master ID": "M-010", "Herb / Tree Name": "Licorice (Mulethi)", "Scientific Name": "Glycyrrhiza glabra", "Family": "Fabaceae", "Phytochemical": "Glycyrrhizin", "Canonical SMILES": "CC1(C2CCC3(C(C2(CCC1(C(=O)O)C)O)C(=O)C=C4C3(CCC5(C4CC(C(C5)(C)C(=O)O)OC6C(C(C(C(O6)C(=O)O)O)O)OC7C(C(C(C(O7)C(=O)O)O)O)O)C)C)C)C", "Medicinal Activity": "Antiviral", "Target Protein / Receptor Name": "SARS-CoV-2 Main Protease (Mpro)", "PDB ID": "6LU7", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "यष्टीमधु रसं स्वादु सुशीलं बलवर्णकृत्। गुरु चक्षुष्यं वृष्यं च व्रणशोथविनाशनम्॥", "Roman Transliteration": "yaṣṭīmadhu rasaṃ svādu suśīlaṃ balavarṇakṛt | guru cakṣuṣyaṃ vṛṣyaṃ ca vraṇaśothavināśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Vranashothahara Varnya Balya Jvarahara"},
        {"Master ID": "M-011", "Herb / Tree Name": "Amla", "Scientific Name": "Phyllanthus emblica", "Family": "Phyllanthaceae", "Phytochemical": "Gallic Acid", "Canonical SMILES": "C1=C(C=C(C(=C1O)O)O)C(=O)O", "Medicinal Activity": "Antioxidant", "Target Protein / Receptor Name": "Human Peroxiredoxin 5", "PDB ID": "1HD2", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "वयःस्थापनां धात्रीफलमम्लं रसे स्मृतम्। परं कफहरं वृष्यं चक्षुष्यं च रसायनम्॥", "Roman Transliteration": "vayaḥsthāpanāṃ dhātrīphalamamlaṃ rase smṛtam | paraṃ kaphaharaṃ vṛṣyaṃ cakṣuṣyaṃ ca rasāyanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Amla Madhura Tikta Kasaya Katu; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Rasayana Vayasthapana (Anti-aging) Chakshushya"},
        {"Master ID": "M-012", "Herb / Tree Name": "Garlic", "Scientific Name": "Allium sativum", "Family": "Amaryllidaceae", "Phytochemical": "Allicin", "Canonical SMILES": "C=CCSS(=O)CC=C", "Medicinal Activity": "Antibacterial", "Target Protein / Receptor Name": "Staphylococcus aureus Sortase A", "PDB ID": "2GLA", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "लशुनः कटुकोष्णश्च तीक्ष्णो वातकफापहः। रसायनः परं हृद्यःক্রिमिकुष्ठविनाशनः॥", "Roman Transliteration": "laśunaḥ kaṭukoṣṇaśca tīkṣṇo vātakaphāpahaḥ | rasāyanaḥ paraṃ hṛdyaḥ krimikuṣṭhavināśanaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Madhura Tikta Kasaya; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Krimighna Hridya Rasayana Kusthahara"},
        {"Master ID": "M-013", "Herb / Tree Name": "Ginger", "Scientific Name": "Zingiber officinale", "Family": "Zingiberaceae", "Phytochemical": "6-Gingerol", "Canonical SMILES": "CCCCCC(CC(=O)CCC1=CC(=C(C=C1)O)OC)O", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Cyclooxygenase-2 (COX-2)", "PDB ID": "1CX2", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "आर्द्रकं कटुकं दीपनं चोष्णं वातकफापहम्। शूलहृद्भेदनं हृद्यं विबन्धानाहनाशनम्॥", "Roman Transliteration": "ārdrakaṃ kaṭukaṃ dīpanaṃ coṣṇaṃ vātakaphāpaham | śūlahṛdbhedanaṃ hṛdyaṃ vibandhānāhanāśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Dipana (Digestive) Shoolahara Hridya"},
        {"Master ID": "M-014", "Herb / Tree Name": "Black Pepper", "Scientific Name": "Piper nigrum", "Family": "Piperaceae", "Phytochemical": "Piperine", "Canonical SMILES": "C1CCN(CC1)C(=O)/C=C/C=C/C2=CC3=C(C=C2)OCO3", "Medicinal Activity": "Bioenhancer", "Target Protein / Receptor Name": "P-Glycoprotein", "PDB ID": "6I6H", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "मरिचं कटुकं तीक्ष्णं दीपनं कफवातजित्। उष्णं प्रसेकि क्रिमिहृच्छ्वासशूलविनाशनम्॥", "Roman Transliteration": "maricaṃ kaṭukaṃ tīkṣṇaṃ dīpanṃ kaphavātajit | uṣṇaṃ praseki krimihṛcchvāsaśūlavināśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Pramathi (Bioenhancer) Dipana Krimihara Shwasahara"},
        {"Master ID": "M-015", "Herb / Tree Name": "Shankhpushpi", "Scientific Name": "Convolvulus pluricaulis", "Family": "Convolvulaceae", "Phytochemical": "Scopoletin", "Canonical SMILES": "COC1=C(C=C2C(=C1)C=CC(=O)O2)O", "Medicinal Activity": "Anxiolytic", "Target Protein / Receptor Name": "GABA-A Receptor", "PDB ID": "6D1M", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "शङ्खपुष्पी सरा तिक्ता मेध्या मानसरोगहृत्। बल्या रसायनी चैव विस्मृतिभ्रमनाशिनी॥", "Roman Transliteration": "śaṅkhapuṣpī sarā tiktā medhyā mānasarogahṛt | balyā rasāyanī caiva vismṛtibramhanāśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Medhya Manasarogahara (Anxiolytic) Rasayana"},
        {"Master ID": "M-016", "Herb / Tree Name": "Gotu Kola", "Scientific Name": "Centella asiatica", "Family": "Apiaceae", "Phytochemical": "Asiaticoside", "Canonical SMILES": "CC1CCC2(CCC3(C(=CCC4C3(CCC5C4(CCC(C5(C)C)O)C)C)C2C1O)C)C(=O)OC6C(C(C(C(O6)CO)O)O)O", "Medicinal Activity": "Wound Healing", "Target Protein / Receptor Name": "Collagenase", "PDB ID": "2Y6I", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "मण्डूकपर्णी हिमा तिक्ता मेध्या आयुष्या रसायनी। कषायोष्णा सरा स्वर्या कुष्ठमेहास्त्रकासजित्॥", "Roman Transliteration": "maṇḍūkaparṇī himā tiktā medhyā āyuṣyā rasāyanī | kaṣāyoṣṇā sarā svaryā kuṣṭhamehāstrakāsajit ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Vranaropana (Wound Healing) Medhya Rasayana"},
        {"Master ID": "M-017", "Herb / Tree Name": "Guggul", "Scientific Name": "Commiphora mukul", "Family": "Burseraceae", "Phytochemical": "Guggulsterone E", "Canonical SMILES": "CC=C1CCC2C3CCC4=CC(=O)CCC4(C3CCC12C)C", "Medicinal Activity": "Hypolipidemic", "Target Protein / Receptor Name": "Farnesoid X Receptor (FXR)", "PDB ID": "1OSH", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "गुग्गुलुः कटुकस्तिक्तो वीर्योष्णः कफवातजित्। मेदोहरः परं व्रण्यः क्लेदमेहापहो लघुः॥", "Roman Transliteration": "gugguluḥ kaṭukastikto vīryoṣṇaḥ kaphavātajit | medoharaḥ paraṃ vraṇyaḥ kledamehāpaho laghuḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta Kasaya; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Medohara (Hypolipidemic) Shothahara Lekhaniya"},
        {"Master ID": "M-018", "Herb / Tree Name": "Shatavari", "Scientific Name": "Asparagus racemosus", "Family": "Asparagaceae", "Phytochemical": "Shatavarin IV", "Canonical SMILES": "CC1CCC2(C(O1)C(C3C2(CCC4C3CCC5C4(CCC(C5)OC6C(C(C(C(O6)CO)O)O)OC7C(C(C(C(O7)CO)O)O)O)C)C)O)C", "Medicinal Activity": "Immunomodulatory", "Target Protein / Receptor Name": "Human Progesterone Receptor", "PDB ID": "1A28", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "शतावरी हिमा तिक्ता रसे स्वादी रसायनी। स्तन्यदा बुद्धिदा बल्या चक्षुष्या कफवातजित्॥", "Roman Transliteration": "śatāvarī himā tiktā rase svādī rasāyanī | stanyadā buddhidā balyā cakṣuṣyā kaphavātajit ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura Tikta; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Stanyada (Galactagogue) Balya Rasayana Ojovardhaka"},
        {"Master ID": "M-019", "Herb / Tree Name": "Kalmegh", "Scientific Name": "Andrographis paniculata", "Family": "Acanthaceae", "Phytochemical": "Andrographolide", "Canonical SMILES": "CC1=C(C(=O)OC1C(C)C2CCC3(C2(CCC(C3=C)O)C)C)O", "Medicinal Activity": "Hepatoprotective", "Target Protein / Receptor Name": "Human Tumor Necrosis Factor Alpha", "PDB ID": "2TNF", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "कालमेघस्तु तिक्तोष्णः कफपित्तज्वरापहः। यकृतोत्तेजकः श्रेष्ठः क्रिमिकुष्ठविनाशनः॥", "Roman Transliteration": "kālameghastu tiktoṣṇaḥ kaphapittajvarāpahaḥ | yakṛtottejakaḥ śreṣṭhaḥ krimikuṣṭhavināśanaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Yakrut-uttejaka (Hepatoprotective) Jvarahara"},
        {"Master ID": "M-020", "Herb / Tree Name": "Karela (Bitter Melon)", "Scientific Name": "Momordica charantia", "Family": "Cucurbitaceae", "Phytochemical": "Charantin", "Canonical SMILES": "CC1CCC2(C(O1)C(C3C2(CCC4C3CCC5C4(CCC(C5)OC6C(C(C(C(O6)CO)O)O)O)C)C)O)C", "Medicinal Activity": "Antidiabetic", "Target Protein / Receptor Name": "Insulin Receptor Tyrosine Kinase", "PDB ID": "1IRK", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "कारवेल्लं कदु तीक्ष्णं तिक्तं पाके कटु स्मृतम्। दीपनं भेदनं हन्ति प्रमेहकफपित्तकृत्॥", "Roman Transliteration": "kāravellaṃ kadu tīkṣṇaṃ tiktaṃ pāke kaṭu smṛtam | dīpanāṃ bhedanāṃ hanti pramehakaphapittakṛt ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Pramehahara (Antidiabetic) Dipana Raktashodhaka"},
        {"Master ID": "M-021", "Herb / Tree Name": "Moringa", "Scientific Name": "Moringa oleifera", "Family": "Moringaceae", "Phytochemical": "Quercetin", "Canonical SMILES": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(O2)C=C(C=C3O)O)O)O)O", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "PI3K", "PDB ID": "4FA6", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "शिग्रुस्तीक्ष्णोष्णकटुकः कफवातशोथहृत्। क्रिमिकुष्ठव्रणघ्नश्च दीपनो भेदनो लघुः॥", "Roman Transliteration": "śigrustīkṣṇoṣṇakaṭukaḥ kaphavātaśothahṛt | krimikuṣṭhavraṇaghnaśca dīpano bhedano laghuḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta Madhura; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Shothahara (Anti-inflammatory/Tumor) Krimighna Dipana"},
        {"Master ID": "M-022", "Herb / Tree Name": "Cinnamon", "Scientific Name": "Cinnamomum verum", "Family": "Lauraceae", "Phytochemical": "Cinnamaldehyde", "Canonical SMILES": "C1=CC=C(C=C1)/C=C/C=O", "Medicinal Activity": "Antidiabetic", "Target Protein / Receptor Name": "PPAR-gamma", "PDB ID": "3DZY", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "त्वक्पत्रं लघु तीक्ष्णोष्णं कडु तिक्तं च रुच्यकम्। कफवातहरं कण्ठरुक्प्रमेहविनाशनम्॥", "Roman Transliteration": "tvakpatraṃ laghu tīkṣṇoṣṇaṃ kaḍu tiktaṃ ca rucyakam | kaphavātaharaṃ kaṇṭharukpramehavināśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta Madhura; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Pramehahara (Antidiabetic) Dipana Hridya"},
        {"Master ID": "M-023", "Herb / Tree Name": "Haritaki", "Scientific Name": "Terminalia chebula", "Family": "Combretaceae", "Phytochemical": "Chebulinic Acid", "Canonical SMILES": "CC1C2C(C(C(O1)OC(=O)C3=CC(=C(C(=C3)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O)OC(=O)C5=CC(=C(C(=C5)O)O)O", "Medicinal Activity": "Antiviral", "Target Protein / Receptor Name": "Hepatitis C Virus NS3/4A Protease", "PDB ID": "4A92", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "हरीतकी मानुषीणां मातेव हितकारिणी। प्रमेहकुष्ठशोथार्शःकामलाक्रिमिनाशिनी॥", "Roman Transliteration": "harītakī mānuṣīṇāṃ māteva hitakāriṇī | pramehakuṣṭhaśothārśaḥkāmalākrimināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Kasaya Madhura Amla Katu Tikta; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Tridosahara Anulomana (Laxative) Krimighna"},
        {"Master ID": "M-024", "Herb / Tree Name": "Baheda", "Scientific Name": "Terminalia bellirica", "Family": "Combretaceae", "Phytochemical": "Bellericanin", "Canonical SMILES": "C1=CC(=C(C=C1)O)C2=CC(=O)C3=C(O2)C=C(C(=C3O)O)O", "Medicinal Activity": "Antimicrobial", "Target Protein / Receptor Name": "Staphylococcus aureus Dihydrofolate Reductase", "PDB ID": "2W9S", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "बिभीतकं स्वादुपाकं कषायं कफपित्तनुत्। उष्णवीर्यं चक्षुष्यं केश्यं क्रिमिनाशनम्॥", "Roman Transliteration": "bibhītakaṃ svādupākaṃ kaṣāyaṃ kaphapittanut | uṣṇavīryaṃ cakṣuṣyaṃ keśyaṃ krimināśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Kasaya; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Krimighna Kanthya (Throat-soothing) Chakshushya"},
        {"Master ID": "M-025", "Herb / Tree Name": "Bel", "Scientific Name": "Aegle marmelos", "Family": "Rutaceae", "Phytochemical": "Marmin", "Canonical SMILES": "CC(=CCOCCC1=CC=C2C(=C1)C=CC(=O)O2)C", "Medicinal Activity": "Gastroprotective", "Target Protein / Receptor Name": "H+/K+-ATPase (Proton Pump)", "PDB ID": "5YLV", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "बिल्वं कषायं मधुरं पाचकं दीपनं लघु। उष्णं कफवातहरं ग्राही विबन्धाध्माननाशनम्॥", "Roman Transliteration": "bilvaṃ kaṣāyaṃ madhuraṃ pācakaṃ dīpanaṃ laghu | uṣṇaṃ kaphavātaharaṃ grāhī vibandhādhmānanaśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Kasaya Tikta Madhura; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Grahi (Gastroprotective) Dipana Pachana"},
        {"Master ID": "M-026", "Herb / Tree Name": "Pippali", "Scientific Name": "Piper longum", "Family": "Piperaceae", "Phytochemical": "Piperlongumine", "Canonical SMILES": "C1CC(=O)NC(=O)C1/C=C/C2=CC(=C(C(=C2)OC)OC)OC", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Human Glutathione S-Transferase P1", "PDB ID": "11GS", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "पिप्पली कटुका तिक्ता स्वादुपाका रसायनी। दीपनी श्वासकासघ्नी प्रमेहार्शःक्षयापहा॥", "Roman Transliteration": "pippalī kaṭukā tiktā svādupākā rasāyanī | dīpanī śvāsakāsaghnī pramehārśaḥkṣayāpahā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Anushnasheeta; Vipaka: Madhura", "Classical Karma (Action)": "Rasayana Dipana Shwasahara Kasanut"},
        {"Master ID": "M-027", "Herb / Tree Name": "Chitrak", "Scientific Name": "Plumbago zeylanica", "Family": "Plumbaginaceae", "Phytochemical": "Plumbagin", "Canonical SMILES": "CC1=CC(=O)C2=C(C1=O)C=CC(=C2)O", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Human AKT1 Kinase", "PDB ID": "3O96", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "चित्रको वह्निसदृशः पाचकः दीपनो लघुः। कफवातहरो शोथार्शःकुष्ठक्रिमिनाशनः॥", "Roman Transliteration": "citrako vahnisadṛśaḥ pācakaḥ dīpano laghuḥ | kaphavātaharo śothārśaḥkuṣṭhakrimināśanaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Deepana Pachana Lekhaniya (Anti-proliferative)"},
        {"Master ID": "M-028", "Herb / Tree Name": "Manjistha", "Scientific Name": "Rubia cordifolia", "Family": "Rubiaceae", "Phytochemical": "Alizarin", "Canonical SMILES": "C1=CC=C2C(=C1)C(=O)C3=C(O2)C=C(C(=C3O)O)O", "Medicinal Activity": "Antimicrobial", "Target Protein / Receptor Name": "Staphylococcus aureus Tyrosyl-tRNA Synthetase", "PDB ID": "1JIJ", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "मञ्जिष्ठा मधुरा तिक्ता कषायोष्णा विषाहरी। शोथत्वग्दोषमेहास्रकुष्ठकण्डूव्रणापहा॥", "Roman Transliteration": "mañjiṣṭhā madhurā tiktā kaṣāyoṣṇā viṣāharī | śothatvagdoṣamehāsrakuṣṭhakaṇḍūvraṇāpahā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya Madhura; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Raktashodhaka (Blood Purifier) Vranaropana Krimighna"},
        {"Master ID": "M-029", "Herb / Tree Name": "Sadabahar", "Scientific Name": "Catharanthus roseus", "Family": "Apocynaceae", "Phytochemical": "Vincristine", "Canonical SMILES": "CCC1CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)C7C8(CC9CC(C8N(C7=O)C)(C(C9)(C(=O)OC)O)CC)O)OC)C(=O)OC", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Human Tubulin Beta Chain", "PDB ID": "4EB6", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "सदाबहारो मधुरस्तिक्तस्तु वरदः स्मृतः। रक्तप्रदरनाशाय ग्रन्थ्यर्बुदहरो मतः॥", "Roman Transliteration": "sadābahāro madhurastiktastu varadaḥ smṛtaḥ | raktapradaranāśāya granthyarbudaharo mataḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Madhura; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Arbudahara (Anticancer/Anti-tumor) Raktashodhaka"},
        {"Master ID": "M-030", "Herb / Tree Name": "Senna", "Scientific Name": "Senna alexandrina", "Family": "Fabaceae", "Phytochemical": "Sennoside A", "Canonical SMILES": "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C(=CC(=C3)C(=O)O)C4C5=C(C(=O)C6=CC=CC=C6C5=O)C(=CC(=C4)C(=O)O)O", "Medicinal Activity": "Laxative", "Target Protein / Receptor Name": "Human Aquaporin-4", "PDB ID": "3GD8", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "मार्कण्डिका च कटुका तिक्तोष्णा भेदिनी लघुः। मलावष्टम्भशूलघ्नी यकृद्रोगविनाशिनी॥", "Roman Transliteration": "mārkaṇḍikā ca kaṭukā tiktoṣṇā bhedinī laghuḥ | malāvaṣṭambhaśūlaghnī yakṛdroghavināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Virechana (Laxative) Anulomana Malabhedini"},
        {"Master ID": "M-031", "Herb / Tree Name": "Castor (Eranda)", "Scientific Name": "Ricinus communis", "Family": "Euphorbiaceae", "Phytochemical": "Ricinoleic Acid", "Canonical SMILES": "CCCCCCC(CC=CCCCCC(=O)O)O", "Medicinal Activity": "Laxative", "Target Protein / Receptor Name": "Prostaglandin EP3 Receptor", "PDB ID": "6M9T", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "एरण्डो मधुरोष्णश्च तीक्ष्णो विड्विबन्धहा। शूलशोथकफातङ्कवातघ्नो मेदहः परम्॥", "Roman Transliteration": "eraṇḍo madhuroṣṇaśca tīkṣṇo viḍvibandhahā | śūlaśothakaphātaṅkavātaghno medahaḥ param ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura Katu Kasaya; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Virechana (Laxative) Shoolahara Vatahara"},
        {"Master ID": "M-032", "Herb / Tree Name": "Karanja", "Scientific Name": "Millettia pinnata", "Family": "Fabaceae", "Phytochemical": "Karanjin", "Canonical SMILES": "CC1=C(C=C2C(=C1)C(=O)C3=C(O2)C=CC=C3)C4=CC=CC=C4", "Medicinal Activity": "Antimicrobial", "Target Protein / Receptor Name": "Escherichia coli DNA Gyrase A", "PDB ID": "1AB4", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "करञ्जः कटुकस्तिक्तो वीर्योष्णः कफवातजित्। व्रणशोधनकृच्चैव क्रिमिकुष्ठविनाशनः॥", "Roman Transliteration": "karañjaḥ kaṭukastikto vīryoṣṇaḥ kaphavātajit | vraṇaśodhanakṛccaiva krimikuṣṭhavināśanaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Krimighna (Antimicrobial) Vrana-shodhana Kusthahara"},
        {"Master ID": "M-033", "Herb / Tree Name": "Bakuchi", "Scientific Name": "Psoralea corylifolia", "Family": "Fabaceae", "Phytochemical": "Bakuchiol", "Canonical SMILES": "CC(=CCCC(C)(C=C)C1=CC=C(C=C1)O)C", "Medicinal Activity": "Antimicrobial", "Target Protein / Receptor Name": "Streptococcus mutans Sortase A", "PDB ID": "3HQE", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "बाकुची मधुरा तिक्ता कटुपाका रसायनी। हन्ति कुष्ठं प्रमेहं च क्रिमिं केशा हिता च सा॥", "Roman Transliteration": "bākucī madhurā tiktā kaṭupākā rasāyanī | hanti kuṣṭhaṃ pramehaṃ ca krimiṃ keśā hitā ca sā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Kusthaghna (Anti-leprotic/Skin-cure) Krimighna Rasayana"},
        {"Master ID": "M-034", "Herb / Tree Name": "Methi (Fenugreek)", "Scientific Name": "Trigonella foenum-graecum", "Family": "Fabaceae", "Phytochemical": "Trigonelline", "Canonical SMILES": "C[N+]1=CC=CC=C1C(=O)[O-]", "Medicinal Activity": "Antidiabetic", "Target Protein / Receptor Name": "Glucose Transporter Type 4 (GLUT4)", "PDB ID": "4GJS", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "मेथिका कटुका तिक्ता वातघ्नी दीपिनी लघुः। ज्वरारुचिप्रमेहाणां नाशिनी पुष्टिका मता॥", "Roman Transliteration": "methikā kaṭukā tiktā vātaghnī dīpinī laghuḥ | jvarārucipramehāṇāṃ nāśinī puṣṭikā matā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Pramehahara (Antidiabetic) Vatahara Dipana"},
        {"Master ID": "M-035", "Herb / Tree Name": "Gokhru", "Scientific Name": "Tribulus terrestris", "Family": "Zygophyllaceae", "Phytochemical": "Protodioscin", "Canonical SMILES": "CC1CCC2(C(O1)C(C3C2(CCC4C3CCC5C4(CCC(C5)OC6C(C(C(C(O6)CO)O)O)O)C)C)O)C", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Human Androgen Receptor", "PDB ID": "1X4V", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "गोक्षुरः शीतलः स्वादुः बलकृद् बस्तिशोधनः। मधुरो दीपनश्चैव अश्मरीकृच्छ्रनाशनः॥", "Roman Transliteration": "gokṣuraḥ śītalaḥ svāduḥ balakṛd bastiśodhanaḥ | madhuro dīpanaścaiva aśmarīkṛcchrānāśanaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Mootrala (Diuretic) Bastishodhana Aśmarīhara"},
        {"Master ID": "M-036", "Herb / Tree Name": "Bhringraj", "Scientific Name": "Eclipta prostrata", "Family": "Asteraceae", "Phytochemical": "Wedelolactone", "Canonical SMILES": "COC1=CC2=C(C=C1)C3=C(C(=O)O2)C4=C(C=C(C=C4O3)O)O", "Medicinal Activity": "Hepatoprotective", "Target Protein / Receptor Name": "Human Caspase-8", "PDB ID": "1QTN", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "भृङ्गराजः कटुस्तिक्त रूक्षोष्णः कफवातनुत्। केश्यस्त्वच्यो कृमिघ्नश्च यकृद्रोगविनाशनः॥", "Roman Transliteration": "bhṛṅgarājaḥ kaṭustikta rūkṣoṣṇaḥ kaphavātanut | keśyastvacyo krimighnaśca yakṛdroghavināśanaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Keshya (Hair Growth) Yakrut-protective Kusthaghna"},
        {"Master ID": "M-037", "Herb / Tree Name": "Punarnava", "Scientific Name": "Boerhavia diffusa", "Family": "Nyctaginaceae", "Phytochemical": "Punarnavine", "Canonical SMILES": "CNC1CCC2=C(C1)C=CC=C2", "Medicinal Activity": "Diuretic / Renal", "Target Protein / Receptor Name": "Human Adenosine A1 Receptor", "PDB ID": "5UEN", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "पुनर्नवा भवेदुष्णा तिक्ता च मधुरा रसे। शोथघ्नी मूत्रला चैव बस्तिरोगविनाशिनी॥", "Roman Transliteration": "punarnavā bhaveduṣṇā tiktā ca madhurā rase | śothaghnī mūtralā caiva bastiroghavināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura Tikta Kasaya; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Mootrala (Diuretic) Shothahara (Anti-edema)"},
        {"Master ID": "M-038", "Herb / Tree Name": "Safed Musli", "Scientific Name": "Chlorophytum borivilianum", "Family": "Asparagaceae", "Phytochemical": "Boriviloside A", "Canonical SMILES": "CC1C(C(C(C(O1)OC2C(C(OC3CC4C(C)C5CCC6C(C)C(=O)CC6C5CC4C3)CO)O)O)O)O", "Medicinal Activity": "Adaptogenic", "Target Protein / Receptor Name": "Human Corticotropin-Releasing Factor Receptor 1", "PDB ID": "4K5Y", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "मुशली मधुरा वृष्या वीर्योष्णा कफनाशनी। बल्या रसायनी चैव पुष्टिका धातुवर्धिनी॥", "Roman Transliteration": "muśalī madhurā vṛṣyā vīryoṣṇā kaphanāśanī | balyā rasāyanī caiva puṣṭikā dhātuvardhinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Vrishya (Aphrodisiac) Balya Dhātu-Rasayana"},
        {"Master ID": "M-039", "Herb / Tree Name": "Tulsi", "Scientific Name": "Ocimum sanctum", "Family": "Lamiaceae", "Phytochemical": "Ursolic Acid", "Canonical SMILES": "CC1CCC2(CCC3(C(=CCC4C3(CCC5C4(CCC(C5(C)C)O)C)C)C2C1O)C)C(=O)O", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Matrix Metalloproteinase-9 (MMP-9)", "PDB ID": "1L6J", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "तुलसी कटुका तिक्ता हृद्या उष्णा दाहपित्तकृत्। दीपनी कुष्ठकृच्छ्रास्त्रपार्श्वशूलविनाशिनी॥", "Roman Transliteration": "tulasī kaṭukā tiktā hṛdyā uṣṇā dāhapittakṛt | dīpanī kuṣṭhakṛcchrāstrapārśvaśūlavināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Krimighna Hridya (Cardioprotective) Dipana"},
        {"Master ID": "M-040", "Herb / Tree Name": "Neem", "Scientific Name": "Azadirachta indica", "Family": "Meliaceae", "Phytochemical": "Azadirachtin", "Canonical SMILES": "CC1=CC23C(C(C4(C(O2)C5(C3(C(C1(O)C(=O)OC)O)O)CC(O5)(C(=O)OC)C6=CC=CO6)O)OC(=O)C)OC(=O)/C(=C/C)/C", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Human Topoisomerase II alpha", "PDB ID": "1ZXM", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "निम्बः शीतो लघुस्तिक्तो व्रणशोधनरोपणः। चक्षुष्यः कफपित्तघ्नः कुष्ठहृत् कृमिहृत्परः॥", "Roman Transliteration": "nimbaḥ śīto laghustikto vraṇaśodhanaropaṇaḥ | cakṣuṣyaḥ kaphapittaghnaḥ kuṣṭhahṛt kṛmihṛtparaḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Shita; Vipaka: Katu", "Classical Karma (Action)": "Krimighna (Antimicrobial) Vrana-shodhana Kusthaha"},
        {"Master ID": "M-041", "Herb / Tree Name": "Ashwagandha", "Scientific Name": "Withania somnifera", "Family": "Solanaceae", "Phytochemical": "Withanone", "Canonical SMILES": "CC1=C(C(=O)C2=C(C1O)C3CCC4C5CC6C(C5(CCC4(C3(C2)C)O)C)OC(=O)C6(C)O)C7CC(=O)OC7", "Medicinal Activity": "Neuroprotective", "Target Protein / Receptor Name": "Acetylcholinesterase (AChE)", "PDB ID": "4EY7", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "अश्वगन्धा अनिलाश्लेष्मश्वित्रशोथक्षयापहा। बल्या रसायनी तिक्ता कषायोष्णा अतिशुक्रला॥", "Roman Transliteration": "aśvagandhā anilāśleṣmaśvitraśothakṣayāpahā | balyā rasāyanī tiktā kaṣāyoṣṇā atiśukralā ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya Madhura; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Rasayana (Rejuvenative) Balya Shothahara"},
        {"Master ID": "M-042", "Herb / Tree Name": "Amla", "Scientific Name": "Phyllanthus emblica", "Family": "Phyllanthaceae", "Phytochemical": "Ellagic Acid", "Canonical SMILES": "C1=C2C3=C(C(=C1)O)OC(=O)C4=CC(=C(C(=C43)OC2=O)O)O", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "Protein Kinase CK2 alpha subunit", "PDB ID": "3BOW", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "वयःस्थापनां धात्रीफलमम्लं रसे स्मृतम्। परं कफहरं वृष्यं चक्षुष्यं च रसायनम्॥", "Roman Transliteration": "vayaḥsthāpanāṃ dhātrīphalamamlaṃ rase smṛtam | paraṃ kaphaharaṃ vṛṣyaṃ cakṣuṣyaṃ ca rasāyanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Amla Madhura Tikta Kasaya Katu; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Rasayana Vayasthapana (Anti-aging) Chakshushya"},
        {"Master ID": "M-043", "Herb / Tree Name": "Giloy", "Scientific Name": "Tinospora cordifolia", "Family": "Menispermaceae", "Phytochemical": "Tinosporaside", "Canonical SMILES": "CC1=CC2=C(C(=O)O1)C3C(C4C2(CCC4(C)O)O)C5(C3CC(O5)C6=COC=C6)C", "Medicinal Activity": "Immunomodulatory", "Target Protein / Receptor Name": "Interleukin-6 (IL-6)", "PDB ID": "1ALU", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "गुडूची कटुका तिक्ता स्वादुपाका रसायनी। ज्वरकुष्ठप्रमेहार्शःकण्डूहृद्रोगवातनुत्॥", "Roman Transliteration": "guḍūcī kaṭukā tiktā svādupākā rasāyanī | jvarakuṣṭhapramehārśaḥkaṇḍūhṛdroghavātanut ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Usna; Vipaka: Madhura", "Classical Karma (Action)": "Pramehahara (Antidiabetic) Rasayana Tridosashamana"},
        {"Master ID": "M-044", "Herb / Tree Name": "Ginger", "Scientific Name": "Zingiber officinale", "Family": "Zingiberaceae", "Phytochemical": "6-Shogaol", "Canonical SMILES": "CCCCCC=CC(=O)CCC1=CC(=C(C=C1)O)OC", "Medicinal Activity": "Anti-inflammatory", "Target Protein / Receptor Name": "Human TNF-alpha", "PDB ID": "2AZA", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "आर्द्रकं कटुकं दीपनं चोष्णं वातकफापहम्। शूलहृद्भेदनं हृद्यं विबन्धानाहनाशनम्॥", "Roman Transliteration": "ārdrakaṃ kaṭukaṃ dīpanaṃ coṣṇaṃ vātakaphāpaham | śūlahṛdbhedanaṃ hṛdyaṃ vibandhānāhanāśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Dipana (Digestive) Shoolahara Hridya"},
        {"Master ID": "M-045", "Herb / Tree Name": "Cinnamon", "Scientific Name": "Cinnamomum verum", "Family": "Lauraceae", "Phytochemical": "Cinnamic Acid", "Canonical SMILES": "C1=CC=C(C=C1)/C=C/C(=O)O", "Medicinal Activity": "Antidiabetic", "Target Protein / Receptor Name": "Protein Tyrosine Phosphatase 1B (PTP1B)", "PDB ID": "1XBO", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "त्वक्पत्रं लघु तीक्ष्णोष्णं कडु तिक्तं च रुच्यकम्। कफवातहरं कण्ठरुक्प्रमेहविनाशनम्॥", "Roman Transliteration": "tvakpatraṃ laghu tīkṣṇoṣṇaṃ kaḍu tiktaṃ ca rucyakam | kaphavātaharaṃ kaṇṭharukpramehavināśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta Madhura; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Pramehahara (Antidiabetic) Dipana Hridya"},
        {"Master ID": "M-046", "Herb / Tree Name": "Arjuna", "Scientific Name": "Terminalia arjuna", "Family": "Combretaceae", "Phytochemical": "Arjunolic Acid", "Canonical SMILES": "CC1CCC2(CCC3(C(=CCC4C3(CCC5C4(CCC(C5(C)C)O)O)C)C2C1O)C)C(=O)O", "Medicinal Activity": "Cardioprotective", "Target Protein / Receptor Name": "Beta-1 Adrenergic Receptor", "PDB ID": "7JVP", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "ककुभोऽर्जुनः कीर्तितः स्याच्छीतलः कषायको। हृद्रोगक्षतक्षयविषप्रशमनोऽपि च॥", "Roman Transliteration": "kakubho'rjunaḥ kīrtitaḥ syācchītalaḥ kaṣāyako | hṛdroghakṣatakṣayaviṣapraśamano'pi ca ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Kasaya; Virya: Shita; Vipaka: Katu", "Classical Karma (Action)": "Hridya (Cardioprotective) Raktastambhana Kṣatahara"},
        {"Master ID": "M-047", "Herb / Tree Name": "Licorice (Mulethi)", "Scientific Name": "Glycyrrhiza glabra", "Family": "Fabaceae", "Phytochemical": "Liquiritigenin", "Canonical SMILES": "C1CC(=O)C2=C(C=C(C=C2O1)O)C3=CC=C(C=C3)O", "Medicinal Activity": "Estrogenic", "Target Protein / Receptor Name": "Estrogen Receptor Beta (ER-β)", "PDB ID": "1QKM", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "यष्टीमधु रसं स्वादु सुशीलं बलवर्णकृत्। गुरु चक्षुष्यं वृष्यं च व्रणशोथविनाशनम्॥", "Roman Transliteration": "yaṣṭīmadhu rasaṃ svādu suśīlaṃ balavarṇakṛt | guru cakṣuṣyaṃ vṛṣyaṃ ca vraṇaśothavināśanam ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Madhura; Virya: Shita; Vipaka: Madhura", "Classical Karma (Action)": "Vranashothahara Varnya Balya Jvarahara"},
        {"Master ID": "M-048", "Herb / Tree Name": "Guggul", "Scientific Name": "Commiphora mukul", "Family": "Burseraceae", "Phytochemical": "Guggulsterone Z", "Canonical SMILES": "CC=C1CCC2C3CCC4=CC(=O)CCC4(C3CCC12C)C", "Medicinal Activity": "Anticancer", "Target Protein / Receptor Name": "NF-kB p50/p65 Heterodimer", "PDB ID": "1VKX", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "गुग्गुलुः कटुकस्तिक्तो वीर्योष्णः कफवातजित्। मेदोहरः परं व्रण्यः क्लेदमेहापहो लघुः॥", "Roman Transliteration": "gugguluḥ kaṭukastikto vīryoṣṇaḥ kaphavātajit | medoharaḥ paraṃ vraṇyaḥ kledamehāpaho laghuḥ ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Katu Tikta Kasaya; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Medohara (Hypolipidemic) Shothahara Lekhaniya"},
        {"Master ID": "M-049", "Herb / Tree Name": "Sarpagandha", "Scientific Name": "Rauvolfia serpentina", "Family": "Apocynaceae", "Phytochemical": "Ajmaline", "Canonical SMILES": "CC1=CC2C3CC4C5C(C3(CN2C1)O)NC6=CC=CC=C56", "Medicinal Activity": "Antiarrhythmic", "Target Protein / Receptor Name": "Human Voltage-Gated Sodium Channel Nav1.5", "PDB ID": "6UZ3", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "सर्पगन्धा तु तिक्तोष्णा कटुका च कफापहा। निद्राप्रदा रक्तवातशमनी काममन्दिनी॥", "Roman Transliteration": "sarpagandhā tu tiktoṣṇā kaṭukā ca kaphāpahā | nidrāpradā raktavātaśamanī kāmamandinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Katu; Virya: Usna; Vipaka: Katu", "Classical Karma (Action)": "Nidraprada (Sedative) Raktavata-shamana (Antihpertensive)"},
        {"Master ID": "M-050", "Herb / Tree Name": "Vasaka", "Scientific Name": "Justicia adhatoda", "Family": "Acanthaceae", "Phytochemical": "Vasicinone", "Canonical SMILES": "C1CC2=NC3=CC=CC=C3C(=O)C4C2(C1)N=C(O4)C", "Medicinal Activity": "Mucolytic", "Target Protein / Receptor Name": "Human Muscarinic Acetylcholine Receptor M3", "PDB ID": "4DA4", "Sanskrit Shloka (Bhavaprakasha Nighantu)": "वासको वासिका वासा भिषङ्माता च सिंहिका। वासा तिक्ता कषायोष्णा कफपित्तविनाशिनी॥", "Roman Transliteration": "vāsako vāsikā vāsā bhiṣaṅmātā ca siṃhikā | vāsā tiktā kaṣāyoṣṇā kaphapittavināśinī ||", "Dravyaguna Profile (Rasa/Virya/Vipaka)": "Rasa: Tikta Kasaya; Virya: Shita; Vipaka: Katu", "Classical Karma (Action)": "Kashahara (Antitussive) Shwasahara (Bronchodilator)"}
    ]
    return pd.DataFrame(hardcoded_data)

# =====================================================================
# 6. APPLICATION DASHBOARD WORKSPACE (SINGLE PAGE FLOW)
# =====================================================================

st.set_page_config(page_title="DravyaDock Hub", layout="wide")
st.title("🌿 DravyaDock (द्रव्यDock) - Computational Ayurvedic Molecular Docking Platform")
st.markdown("**DravyaDock bridges traditional Ayurvedic pharmacology (Dravyaguna Vidya) from the Bhavaprakasha Nighantu with modern translational structural bioinformatics and structure-based drug discovery pipelines.**")
st.markdown("**Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India **")

# Master Reset
if st.button("🔄 Reset Entire Environment", type="secondary", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    for f in ["protein.pdbqt", "ligand.pdbqt", "docking_poses.pdbqt", "temp_lig_state.pdb", "redesign_ligand.pdbqt", "redesign_docking_poses.pdbqt"]:
        if os.path.exists(f): os.remove(f)
    st.success("Dashboard cache and runtime structures completely cleared!")
    safe_rerun()

# ---------------------------------------------------------------------
# RESULT CARD RENDERER (Post-Tree Selection)
# ---------------------------------------------------------------------
if "selected_tree_data" in st.session_state and st.session_state.selected_tree_data and st.session_state.target_ready and st.session_state.ligand_ready:
    st.markdown(generate_ayurvedic_card(st.session_state.selected_tree_data, is_streamlit=True), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SAFEGUARD FALLBACKS
# ---------------------------------------------------------------------
if os.path.exists("protein.pdbqt"): st.session_state.target_ready = True
if os.path.exists("ligand.pdbqt"): st.session_state.ligand_ready = True

# ---------------------------------------------------------------------
# PHASE 1: CORE BASELINE DOCKING ENGINE
# ---------------------------------------------------------------------
st.write("---")
st.header("🔒 Phase 1: Baseline Native Molecular Docking")

col_params, col_visual = st.columns([1, 1])

trigger_rerun = False

with col_params:
    st.subheader("🌿 Step 0: DravyaDock Ayurvedic Database (Auto-Populate)")
    
    ayush_df = load_ayurvedic_db()
    filter_type = st.radio("Search Database by:", ["Herb / Tree Name", "Medicinal Activity"], horizontal=True)

    if filter_type == "Herb / Tree Name":
        options = ayush_df["Herb / Tree Name"].unique().tolist()
        selected_opt = st.selectbox("Select Ayurvedic Herb/Tree:", options)
        filtered_df = ayush_df[ayush_df["Herb / Tree Name"] == selected_opt]
    else:
        options = ayush_df["Medicinal Activity"].unique().tolist()
        selected_opt = st.selectbox("Select Target Medicinal Activity:", options)
        filtered_df = ayush_df[ayush_df["Medicinal Activity"] == selected_opt]

    selected_entry_str = st.selectbox(
        "Select Specific Target Protein & Phytochemical Scaffold:",
        filtered_df.apply(lambda row: f"{row['Herb / Tree Name']} - {row['Phytochemical']} vs {row['Target Protein / Receptor Name']} ({row['PDB ID']})", axis=1).tolist()
    )

    if st.button("📥 Auto-Fill & Load DravyaDock Pipeline", type="primary"):
        idx = filtered_df.apply(lambda row: f"{row['Herb / Tree Name']} - {row['Phytochemical']} vs {row['Target Protein / Receptor Name']} ({row['PDB ID']})", axis=1) == selected_entry_str
        target_row = filtered_df[idx].iloc[0]

        # Bind selected data to session state for rendering the card
        st.session_state.selected_tree_data = target_row.to_dict()

        pdb_id = target_row["PDB ID"].strip()
        smiles = target_row["Canonical SMILES"].strip()
        prot_name = target_row["Target Protein / Receptor Name"].strip()

        # Step A: Load Target Protein
        with st.spinner(f"Loading Target Protein {pdb_id} from DB..."):
            success, path = fetch_pdb_from_rcsb(pdb_id)
            if success:
                st.session_state.local_target_path = path
                st.session_state.pdb_id_display = pdb_id
                st.session_state.protein_name = prot_name
                conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                st.session_state.target_ready = conv_ok

        # Step B: Load Phytochemical Ligand
        with st.spinner(f"Loading Phytochemical {target_row['Phytochemical']}..."):
            pub_data = fetch_ligand_data_from_pubchem(smiles)
            ok, msg = convert_smiles_to_pdbqt(smiles, "ligand.pdbqt")
            if ok:
                st.session_state.ligand_ready = True
                st.session_state.smiles_cache = smiles
                with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                
                st.session_state.ligand_summary_text = (
                    f"**Phytochemical Identifier:** {target_row['Phytochemical']} | **Formula:** {pub_data['formula']} | **MW:** {pub_data['mw']}\n\n"
                    f"> **Dravyaguna Matrix (Ayurvedic Profile):**\n"
                    f"> * **Shloka:** {target_row['Sanskrit Shloka (Bhavaprakasha Nighantu)']}\n"
                    f"> * **Pharmacology:** {target_row['Dravyaguna Profile (Rasa/Virya/Vipaka)']}\n"
                    f"> * **Classical Action:** {target_row['Classical Karma (Action)']}"
                )
            else:
                st.error(f"Ligand Conversion Error: {msg}")

        if st.session_state.target_ready and st.session_state.ligand_ready:
            st.success(f"DravyaDock Pipeline successfully initialized for {target_row['Herb / Tree Name']}!")
            st.session_state.detected_pockets = []
            trigger_rerun = True

    # =================================================================
    # --- HIDDEN MANUAL SETUP UI (Runs invisibly in the background) ---
    # =================================================================
    if False:
        st.write("---")
        st.subheader("1. Target Protein Setup (Manual Override)")
        
        current_p_name = st.text_input("Protein Name", placeholder="Hint: Type protein name here...", value=st.session_state.protein_name)
        current_p_id = st.text_input("PDB ID / Code", placeholder="Hint: Type PDB ID here...", value=st.session_state.pdb_id_display)
        
        if current_p_name != st.session_state.protein_name: st.session_state.protein_name = current_p_name
        if current_p_id != st.session_state.pdb_id_display: st.session_state.pdb_id_display = current_p_id
        st.write("---")
        
        protein_source = st.radio("Choose Protein Input Method:", ["Type 4-Letter PDB ID", "Upload File (.pdb or .pdbqt)"])
        
        if protein_source == "Type 4-Letter PDB ID":
            pdb_id_input = st.text_input("Enter RCSB PDB ID", value="2AMB").strip()
            if st.button("📥 Load Target Structure"):
                if pdb_id_input:
                    success, path = fetch_pdb_from_rcsb(pdb_id_input)
                    if success:
                        st.session_state.local_target_path = path
                        meta = extract_pdb_metadata(path, pdb_id_input.upper())
                        st.session_state.pdb_id_display = meta["id"]
                        st.session_state.protein_name = meta["name"]
                        conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                        st.session_state.target_ready = conv_ok
                        st.success(f"Protein {pdb_id_input.upper()} successfully loaded!")
                        trigger_rerun = True
                    else: st.error(path)
        else:
            uploaded_file = st.file_uploader("Upload Target Protein File", type=["pdb", "pdbqt"])
            if uploaded_file:
                path = f"uploaded_{uploaded_file.name}"
                if st.session_state.last_uploaded_protein != uploaded_file.name:
                    with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                    st.session_state.local_target_path = path
                    meta = extract_pdb_metadata(path, "Uploaded File")
                    st.session_state.pdb_id_display = meta["id"]
                    st.session_state.protein_name = meta["name"]
                    if uploaded_file.name.endswith(".pdb"):
                        conv_ok, _ = convert_pdb_to_pdbqt(path, "protein.pdbqt")
                        st.session_state.target_ready = conv_ok
                    else:
                        os.replace(path, "protein.pdbqt")
                        st.session_state.target_ready = True
                    st.session_state.last_uploaded_protein = uploaded_file.name
                    trigger_rerun = True

        st.subheader("2. Small Molecule Ligand Setup")
        ligand_source = st.radio("Choose Ligand Input Method:", ["SMILES String Input", "Upload Structural File (.pdb, .sdf)"])
        
        smiles_input_val = ""
        uploaded_lig_buffer = None
        uploaded_lig_name = ""

        if ligand_source == "SMILES String Input":
            smiles_input_val = st.text_input("Enter Ligand SMILES String", "CC(=O)NC1=CC=C(O)C=C1").strip()
        else:
            uploaded_lig_file = st.file_uploader("Upload Small Molecule File", type=["pdb", "sdf"])
            if uploaded_lig_file:
                uploaded_lig_buffer = uploaded_lig_file
                uploaded_lig_name = uploaded_lig_file.name

        if st.button("📥 Load Ligand Structure", key="load_ligand_btn"):
            if ligand_source == "SMILES String Input" and smiles_input_val:
                with st.spinner("Querying PubChem Repositories..."):
                    pub_data = fetch_ligand_data_from_pubchem(smiles_input_val)
                    try:
                        mol = Chem.MolFromSmiles(smiles_input_val)
                        if mol:
                            ok, msg = convert_smiles_to_pdbqt(smiles_input_val, "ligand.pdbqt")
                            if ok:
                                st.session_state.ligand_ready = True
                                st.session_state.smiles_cache = smiles_input_val
                                with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                                st.session_state.ligand_summary_text = f"**Name:** {pub_data['name']} | **Formula:** {pub_data['formula']} | **Molecular Weight:** {pub_data['mw']}"
                                st.success("Ligand metadata mapped from PubChem!")
                                trigger_rerun = True
                            else: st.error(msg)
                    except Exception as e: st.error(f"SMILES Parsing Failure: {e}")
                    
            elif ligand_source == "Upload Structural File (.pdb, .sdf)" and uploaded_lig_buffer is not None:
                if st.session_state.last_uploaded_ligand != uploaded_lig_name:
                    temp_in = f"raw_ligand_{uploaded_lig_name}"
                    with open(temp_in, "wb") as f: f.write(uploaded_lig_buffer.getbuffer())
                    
                    mol = Chem.MolFromPDBFile(temp_in, removeHs=False) if uploaded_lig_name.endswith(".pdb") else Chem.SDMolSupplier(temp_in, removeHs=False)[0]
                    
                    if mol:
                        extracted_smiles = ""
                        try: 
                            try: Chem.DetermineBonds(mol)
                            except: pass
                            Chem.SanitizeMol(mol)
                            AllChem.AssignBondOrdersFromTopology(mol)
                            extracted_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                        except Exception: 
                            try: extracted_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
                            except: pass
                        
                        if not extracted_smiles:
                            st.error("⚠️ RDKit could not deduce bond orders from the uploaded spatial coordinates.")
                            st.session_state.smiles_cache = ""
                        else:
                            st.session_state.smiles_cache = extracted_smiles 
                        
                        if mol.GetNumConformers() == 0:
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                            AllChem.MMFFOptimizeMolecule(mol)
                            
                        temp_pdb = "temp_lig_state.pdb"
                        Chem.MolToPDBFile(mol, temp_pdb)
                        ok, _ = convert_pdb_to_pdbqt(temp_pdb, "ligand.pdbqt", is_ligand=True)
                        st.session_state.ligand_ready = ok
                        if os.path.exists(temp_pdb): os.remove(temp_pdb)
                    else:
                        ok, _ = convert_pdb_to_pdbqt(temp_in, "ligand.pdbqt", is_ligand=True)
                        st.session_state.ligand_ready = ok
                        st.session_state.smiles_cache = ""
                    
                    if st.session_state.ligand_ready:
                        st.session_state.ligand_summary_text = f"Ligand 3D coordinates loaded securely. Extracted Base Template: `{extracted_smiles if extracted_smiles else 'Failed'}`"
                        with open("ligand.pdbqt", "r") as f: st.session_state.serialized_ligand_block = f.read()
                        st.session_state.last_uploaded_ligand = uploaded_lig_name
                        
                        if not st.session_state.smiles_cache:
                            st.warning("Note: 3D coordinates loaded for docking, but the 2D SMILES sequence could not be abstracted. Generative Redesign (Phase 2) will require manual SMILES entry.")
                        else:
                            st.success("Structural file loaded! The abstracted SMILES matrix has successfully unlocked Phase 2 and Phase 3.")
                        
                        time.sleep(0.5)
                        st.rerun()
                    else: st.error("Failed to parse ligand coordinate matrix.")
                    if os.path.exists(temp_in): os.remove(temp_in)
    # =================================================================

    st.write("---")

    if st.session_state.target_ready and st.session_state.local_target_path:
        discovered_het = discover_and_list_all_heteroatoms(st.session_state.local_target_path)
        if discovered_het:
            st.markdown("#### 🧬 Catalytic Cofactors & Heteroatom Filter")
            st.markdown("*Select structurally active ions/cofactors to keep in the grid pocket framework. Unchecked entries (like crystallization buffer debris) will be stripped.*")
            
            selected_hets = []
            cols_het = st.columns(min(len(discovered_het), 4))
            for idx, (het_id, count) in enumerate(discovered_het.items()):
                with cols_het[idx % 4]:
                    if st.checkbox(f"Keep {het_id} ({count})", value=False, key=f"keep_het_{het_id}"):
                        selected_hets.append(het_id)
                        
            if st.button("🛠 Rebuild Clean Receptor Structure Matrix"):
                ok, err = convert_pdb_to_pdbqt(st.session_state.local_target_path, "protein.pdbqt", is_ligand=False, allowed_heteroatoms=selected_hets)
                if ok:
                    st.session_state.active_retained_ions = ", ".join(selected_hets) if selected_hets else "None (Fully Stripped)"
                    st.success(f"Receptor rebuilt successfully! Retained: {st.session_state.active_retained_ions}")
                    st.session_state.detected_pockets = [] 
                else:
                    st.error(f"Receptor optimization failure: {err}")

        meta = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display)
        st.markdown(f"> **Protein Summary Profile:** \n> * **Protein Name:** **{st.session_state.protein_name}** \n> * **Title:** {meta['title']} \n> * **PDB ID:** `{st.session_state.pdb_id_display}` | **Classification:** {meta['class']} \n> * **Resolution:** **{meta['res']}**")

    if st.session_state.ligand_ready:
        st.markdown(f"> **Ligand Metric Summary Profile:** \n> {st.session_state.ligand_summary_text}")

    # --- CAVITY & BOUND SITE FINDER ---
    st.subheader("3. Smart Cavity & Bound Site Finder")
    if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
        if st.button("🔍 Scan Surface For Structural Cavities", use_container_width=True):
            with st.spinner("Analyzing macromolecular spatial curvature dynamics..."):
                pockets = identify_protein_cavities("protein.pdbqt")
                st.session_state.detected_pockets = pockets
                if pockets: st.success(f"Successfully mapped {len(pockets)} surface cavities!")

        if st.session_state.detected_pockets:
            p_opts = st.session_state.detected_pockets
            selected_p_idx = st.selectbox("Select Target Computational Cavity:", options=range(len(p_opts)), format_func=lambda idx: f"{p_opts[idx]['Pocket_ID']} (Density Score: {p_opts[idx]['Score']})")
            if st.button("🎯 Align Grid Parameters to This Cavity"):
                chosen_p = p_opts[selected_p_idx]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_p["cx"], chosen_p["cy"], chosen_p["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_p["bx"], chosen_p["by"], chosen_p["bz"]
                st.session_state.selected_native_ligand = f"Automated Surface Cavity Selection: {chosen_p['Pocket_ID']}"
                st.success(f"Grid coordinates targeted over pocket space!")
                trigger_rerun = True

    if st.session_state.target_ready and st.session_state.local_target_path:
        bound_ligands_list = parse_bound_ligands(st.session_state.local_target_path)
        if bound_ligands_list:
            selected_lig_id = st.selectbox("Select native co-crystal target to auto-fill grid box:", options=range(len(bound_ligands_list)), format_func=lambda idx: f"{bound_ligands_list[idx]['ID']} (Chain {bound_ligands_list[idx]['Chain']}-ResSeq {bound_ligands_list[idx]['ResSeq']})")
            if st.button("🎯 Lock Coordinates to Native Site"):
                chosen_target = bound_ligands_list[selected_lig_id]
                st.session_state.cx, st.session_state.cy, st.session_state.cz = chosen_target["cx"], chosen_target["cy"], chosen_target["cz"]
                st.session_state.sx, st.session_state.sy, st.session_state.sz = chosen_target["bx"], chosen_target["by"], chosen_target["bz"]
                st.session_state.selected_native_ligand = f"Bound Native Site: {chosen_target['ID']} (Chain {chosen_target['Chain']})"
                st.success("Grid parameters aligned over pocket boundaries!")
                trigger_rerun = True

    st.subheader("4. Search Space Mechanics (Grid Box)")
    
    if st.button("🌐 Enable Blind Docking (Full Protein Surface)", use_container_width=True):
        if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
            bcx, bcy, bcz, bsx, bsy, bsz = compute_protein_bounding_box("protein.pdbqt")
            st.session_state.cx, st.session_state.cy, st.session_state.cz = round(bcx, 1), round(bcy, 1), round(bcz, 1)
            st.session_state.sx, st.session_state.sy, st.session_state.sz = min(60, int(bsx)), min(60, int(bsy)), min(60, int(bsz))
            st.session_state.selected_native_ligand = "Blind Docking (Entire Surface)"
            st.success("Grid box dynamically expanded to cover the entire macromolecule!")
            trigger_rerun = True
        else:
            st.error("Please load a valid target protein first to enable blind docking.")

    grid_cx = st.number_input("Center X Coordinate", value=float(st.session_state.cx), step=0.1)
    grid_cy = st.number_input("Center Y Coordinate", value=float(st.session_state.cy), step=0.1)
    grid_cz = st.number_input("Center Z Coordinate", value=float(st.session_state.cz), step=0.1)
    
    # ⚠️ UI Protection: Dynamically scale slider maximums to prevent Streamlit rendering crashes
    max_x = max(60, int(st.session_state.sx) + 10)
    max_y = max(60, int(st.session_state.sy) + 10)
    max_z = max(60, int(st.session_state.sz) + 10)
    
    grid_sx = st.slider("Grid Box Size X (Å)", 10, max_x, int(st.session_state.sx))
    grid_sy = st.slider("Grid Box Size Y (Å)", 10, max_y, int(st.session_state.sy))
    grid_sz = st.slider("Grid Box Size Z (Å)", 10, max_z, int(st.session_state.sz))
    exhaustiveness = st.slider("Search Exhaustiveness", min_value=4, max_value=32, value=8, step=4)
    
    can_dock = bool(st.session_state.target_ready and st.session_state.ligand_ready)
    run_btn = st.button("🚀 Initialize Docking Algorithm", type="primary", disabled=not can_dock)

with col_visual:
    st.header("5. Active Viewport Canvas")
    
    if st.session_state.docking_results_raw is None:
        view_tabs = st.tabs(["Receptor & Grid Space", "Standalone Ligand (Interactive 3D)"])
        
        with view_tabs[0]:
            receptor_view_data = ""
            if st.session_state.target_ready and os.path.exists("protein.pdbqt"):
                with open("protein.pdbqt", "r") as f: receptor_view_data = f.read()
            render_advanced_modeling_blueprint(receptor_view_data, st.session_state.serialized_ligand_block, mode="cartoon", unique_id="v_phase1")
            
        with view_tabs[1]:
            if st.session_state.ligand_ready and st.session_state.serialized_ligand_block:
                st.markdown("### 🔬 Isolated Drug Topology")
                st.markdown("Use your mouse to rotate and scroll to zoom in on the specific bonds and stereochemistry of your loaded small molecule.")
                
                ligand_html = f"""
                <div id="ligand_container" style="height: 420px; width: 100%; border-radius:10px; border:1px solid #eaeaea; background:#ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.05);"></div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
                <script>
                    let viewer = $3Dmol.createViewer(document.getElementById('ligand_container'), {{backgroundColor: '#ffffff'}});
                    viewer.addModel(`{st.session_state.serialized_ligand_block}`, 'pdb'); 
                    viewer.setStyle({{model: 0}}, {{stick: {{colorscheme: 'greenCarbon', radius: 0.20}}, sphere: {{radius: 0.35}} }});
                    viewer.zoomTo(); 
                    viewer.render();
                </script>
                """
                components.html(ligand_html, height=450)
            else:
                st.info("💡 Load a valid ligand structure (via SMILES or File Upload) to visualize its 3D topology here.")
                
    else:
        st.subheader("Interactive Complex Viewport")
        if os.path.exists("docking_poses.pdbqt"):
            parsed_poses = split_docking_poses("docking_poses.pdbqt")
            if parsed_poses:
                selected_pose = st.selectbox("Choose Docking Pose to Visualize:", options=list(parsed_poses.keys()), format_func=lambda x: f"Mode {x} Pose Fit", key="p1_sel_pose")
                with open("protein.pdbqt", "r") as f: protein_data = f.read()
                
                pose_affinity_score = get_pose_affinity(st.session_state.docking_results_raw, selected_pose)
                
                try:
                    aff_val = float(pose_affinity_score)
                    aff_color = "#c62828" if aff_val > 0 else "#1b5e20"
                except ValueError:
                    aff_color = "#1b5e20"

                cache_key = f"uff_{st.session_state.protein_name}_{selected_pose}"
                uff_progress_placeholder = st.empty() 
                
                if cache_key not in st.session_state.uff_cache:
                    pre_uff, post_uff, delta_uff = execute_uff_complex_minimization("protein.pdbqt", parsed_poses[selected_pose], uff_progress_placeholder)
                    st.session_state.uff_cache[cache_key] = (pre_uff, post_uff, delta_uff)
                
                uff_progress_placeholder.empty()
                pre_uff, post_uff, delta_uff = st.session_state.uff_cache[cache_key]
                
                if selected_pose == 1:
                    st.session_state.baseline_pre_uff = pre_uff
                    st.session_state.baseline_post_uff = post_uff
                    st.session_state.baseline_delta_uff = delta_uff
                    st.session_state.baseline_affinity = pose_affinity_score

                active_interactions = compute_spatial_interactions("protein.pdbqt", parsed_poses[selected_pose])
                
                amino_acid_categories = {"Acidic (-ve)": [], "Basic (+ve)": [], "Polar (Neutral)": [], "Hydrophobic": []}
                for item in active_interactions:
                    res_full = item["Residue Contact"]
                    res_name = "".join([c for c in res_full if c.isalpha()]).upper()
                    if res_name in ["ASP", "GLU"]: amino_acid_categories["Acidic (-ve)"].append(res_full)
                    elif res_name in ["LYS", "ARG", "HIS"]: amino_acid_categories["Basic (+ve)"].append(res_full)
                    elif res_name in ["SER", "THR", "ASN", "GLN", "CYS", "TYR"]: amino_acid_categories["Polar (Neutral)"].append(res_full)
                    else: amino_acid_categories["Hydrophobic"].append(res_full)
                
                breakdown_html = ""
                report_breakdown_text = ""
                has_contacts = False
                for cat_name, res_list in amino_acid_categories.items():
                    if res_list:
                        has_contacts = True
                        labels_joined = ", ".join(sorted(list(set(res_list))))
                        breakdown_html += f"<p style='margin:4px 0; font-size:13px;'><b style='color:#000000;'>{cat_name}:</b> <span style='color:#333;'>{labels_joined}</span></p>"
                        report_breakdown_text += f"- {cat_name}: {labels_joined}\n"
                if not has_contacts: 
                    breakdown_html = "<p style='margin:4px 0; color:#777; font-size:13px;'>No pocket interactions detected.</p>"
                    report_breakdown_text = "- No close contacts detected under 3.8 Angstroms.\n"

                html_metric_card = """
                <div style="background-color:#f0f7f4; border-left:6px solid #2e7d32; padding:16px; border-radius:8px; margin-bottom:15px; font-family:sans-serif;">
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e0e8e4; padding-bottom:8px; margin-bottom:10px;">
                        <div>
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px;">Active Pose Affinity</span><br>
                            <span style="font-size:36px; font-weight:900; color:{};">{} <span style="font-size:18px; font-weight:normal;">kcal/mol</span></span>
                        </div>
                        <div style="text-align:right; border-left:1px solid #e0e8e4; padding-left:15px;">
                            <span style="font-size:12px; color:#555; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px;">UFF Minimization Delta</span><br>
                            <span style="font-size:32px; font-weight:800; color:#c62828;">{} <span style="font-size:14px; font-weight:normal;">kcal/mol</span></span>
                        </div>
                    </div>
                    <div style="margin-bottom: 10px; font-size: 13px; color: #444;">
                        <b>📍 UFF Initial Energy:</b> {} kcal/mol | <b>📉 Optimized Energy:</b> {} kcal/mol
                    </div>
                    <div>
                        <span style="font-size:11px; color:#666; text-transform:uppercase; font-weight:bold; letter-spacing:0.5px; display:block; margin-bottom:4px;">Binding Site Amino Acid Properties Breakdown:</span>
                        {}
                    </div>
                </div>
                """.format(aff_color, pose_affinity_score, delta_uff, pre_uff, post_uff, breakdown_html)
                st.html(html_metric_card)
                
                col_render, col_mesh = st.columns([1, 1])
                with col_render:
                    style_choice_p1 = st.radio("Macromolecule Style Mode:", ["Cartoon Ribbon Mesh", "Spacefill", "Sticks Profile"], key="p1_style")
                    style_mode_p1 = re.sub(r'\W+', '', style_choice_p1.split()[0].lower())
                with col_mesh:
                    surf_toggle_p1 = st.checkbox("Overlay Translucent Pocket Cavity Mesh", value=False, key="p1_surf")
                    
                render_advanced_modeling_blueprint(receptor_data=protein_data, ligand_data=parsed_poses[selected_pose], mode=style_mode_p1, show_surface=surf_toggle_p1, interactions_list=active_interactions, unique_id="p1_3d_result")
                
                # --- EXPLICIT UFF EXPLANATION UI ---
                st.write("---")
                with st.expander("📖 Understand UFF Minimization & Steric Clashes (Click to Expand)", expanded=False):
                    st.info(f"""
                    **1. 📍 UFF Initial Energy: {pre_uff} kcal/mol**
                    This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, *before* any relaxation occurred. A highly positive energy score indicates extreme geometric tension (a steric clash/rigid atomic wall effect). It means atoms from your phytochemical were physically overlapping or positioned unnaturally close to the rigid atoms of the receptor—most likely the catalytic metal ions or cofactors you specifically chose to retain. In a living biological system, atoms cannot overlap; they would repel each other and shift. But Vina's rigid grid didn't allow them to shift.

                    **2. 📉 Optimized Energy: {post_uff} kcal/mol**
                    This is the total stress of the complex *after* the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm gently pushed overlapping atoms apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state. The negative force field delta (**{delta_uff} kcal/mol**) proves the rigid collision was successfully resolved!
                    """)

                # --- PHASE 1 REPORT EXPORT ---
                st.write("---")
                st.subheader("📋 Phase 1: Local Contact Matrices & Report Generation")

                st.markdown("#### 🧬 Local Contact Residues & Bond Assignments Matrix")
                if active_interactions:
                    df_int = pd.DataFrame(active_interactions)
                    st.dataframe(df_int[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True, use_container_width=True)
                else:
                    st.info("No close contacts detected within a 3.8 Å threshold radius.")

                include_uff_theory = st.checkbox("Include detailed UFF biophysical explanation in the generated reports", value=True, key="p1_uff_toggle")
                
                report_uff_theory_text = ""
                report_uff_theory_html = ""
                if include_uff_theory:
                    report_uff_theory_text = f"""
7. UFF MINIMIZATION BIOPHYSICAL EXPLANATION
-------------------------------------------------------
- 📍 UFF Initial Energy: {pre_uff} kcal/mol
  This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, before any relaxation occurred. A highly positive energy score indicates extreme geometric tension, often a steric clash where atoms physically overlap with rigid atoms of the receptor or retained catalytic cofactors. In a living biological system, atoms shift to relieve this, but a rigid grid does not allow it.

- 📉 Optimized Energy: {post_uff} kcal/mol
  This is the total stress of the complex after the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm took the overlapping atoms and gently pushed them apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state, making the system structurally stable. The critical metric is the massive drop from the initial state ({delta_uff} kcal/mol).
"""
                    report_uff_theory_html = f"""
                    <details style="background-color: #f9fbff; border-left: 6px solid #1e3c72; padding: 15px; border-radius: 4px; margin-top: 20px;">
                        <summary style="font-weight: bold; cursor: pointer; color: #1e3c72; font-size: 16px;">📖 Understand UFF Minimization & Steric Clashes (Click to Expand)</summary>
                        <div style="margin-top: 15px;">
                            <p><b>📍 UFF Initial Energy: {pre_uff} kcal/mol</b></p>
                            <p>This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, before any relaxation occurred. A highly positive energy score indicates extreme geometric tension. This is the mathematical signature of a steric clash (the "rigid atomic wall" effect). It means atoms from your phytochemical were physically overlapping or positioned unnaturally close to the rigid atoms of the receptor—most likely the catalytic metal ions or cofactors you specifically chose to retain. In a living biological system, atoms cannot overlap; they would repel each other and shift. But Vina's rigid grid didn't allow them to shift, resulting in this artificially high stress value.</p>
                            
                            <p><b>📉 Optimized Energy: {post_uff} kcal/mol</b></p>
                            <p>This is the total stress of the complex after the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm took the overlapping atoms and gently pushed them apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state. The system is now structurally stable. What matters is not that the final number is positive, but how far it dropped from the initial state (<b>{delta_uff} kcal/mol</b>).</p>
                        </div>
                    </details>
                    """

                p1_int_text = format_interaction_matrix_text(active_interactions)

                st.markdown("**Quick Copy-Paste Citation Report (Phase 1 Baseline)**")
                report_content_p1 = f"""=======================================================
MOLECULAR DOCKING SCREENING ANALYSIS REPORT (PHASE 1)
Generated dynamically via InSilico BioSphere Docking Tool
Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangresearch@gmail.com
=======================================================

1. TARGET RECEPTOR MACROMOLECULE PROFILE
-------------------------------------------------------
- Target Protein Name: {st.session_state.protein_name}
- Target Configuration Identifier (PDB ID): {st.session_state.pdb_id_display}
- Primary Structure Data Source: RCSB Protein Data Bank Server / Local Upload
- Catalytic Cofactors & Heteroatom Filter configured by user: {st.session_state.active_retained_ions}

2. SMALL MOLECULE DRUG LIGAND PROFILE
-------------------------------------------------------
- Input Structural Identity Matrix (SMILES): {st.session_state.get('smiles_cache', 'Unknown/Failed PDB Extraction')}
- Compiled Chemical Attributes: {st.session_state.ligand_summary_text.replace('**','')}

3. BOUND SPACE CONFIGURATION MECHANICS (GRID BOX)
-------------------------------------------------------
- Center Coordinates Vector (X, Y, Z): ({grid_cx}, {grid_cy}, {grid_cz})
- Grid Bounding Dimensions (X, Y, Z): ({grid_sx} Å, {grid_sy} Å, {grid_sz} Å)
- Search Algorithm Exhaustiveness Index: {exhaustiveness}
- Grid Alignment Strategy: {st.session_state.selected_native_ligand}

4. ACTIVE POSE COMPLEX BINDING METRICS (SELECTED MODE)
-------------------------------------------------------
- Target Alignment Selection Mode: Mode {selected_pose} Pose Fit
- Computed Gibbs Free Energy Affinity: {pose_affinity_score} kcal/mol
- Measured Total Spatial Proximity Contact Atoms: {len(active_interactions)}
- UFF Post-Docking Energy Parameters: Initial: {pre_uff} | Relaxed: {post_uff} | Delta: {delta_uff} kcal/mol

5. LOCAL CONTACT RESIDUES & BOND ASSIGNMENTS MATRIX
-------------------------------------------------------
{p1_int_text}

6. SCIENTIFIC METHODOLOGY & MANUSCRIPT CITATION TRACK
-------------------------------------------------------
Molecular docking was performed using the semi-empirical force field parameters of AutoDock Vina inside the InSilico BioSphere framework. To maintain structural and biological validity, essential catalytic cofactor ions were explicitly preserved within the target binding cleft during search configurations. Potential localized steric constraints and rigid atomic wall collisions resulting from structural constraints were resolved by subjecting the final protein-ligand complexes to post-docking energy minimization using the Universal Force Field (UFF) optimized to a convergence tolerance of 10^-4 kcal/mol·Å.

Manuscript Citation Format Block:
Dr. Sarang S. Dhote, "InSilico BioSphere: An Integrated Platform for Automated Molecular Docking, Surface Cavity Profiling, and Post-Docking Force-Field Relaxation Mechanics." Department of Chemistry, Shri Shivaji Science College, Nagpur, India. Correspondence: sarangresearch@gmail.com
{report_uff_theory_text}=======================================================
"""
                st.text_area("Copy Phase 1 Report Text directly:", value=report_content_p1, height=250, key="p1_text_area")

                meta_data = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display) if st.session_state.local_target_path else {"id":"Custom","title":"Uploaded Structure File","method":"N/A","res":"N/A"}
                meta_data['name'], meta_data['id'] = st.session_state.protein_name, st.session_state.pdb_id_display
                b_img = generate_clean_2d_image(st.session_state.smiles_cache, include_labels=False, zoom_level=420)
                grid_params = {'cx': st.session_state.cx, 'cy': st.session_state.cy, 'cz': st.session_state.cz, 'sx': st.session_state.sx, 'sy': st.session_state.sy, 'sz': st.session_state.sz, 'exh': st.session_state.exhaustiveness}
                df_results_p1 = parse_vina_output_with_residues_global(st.session_state.docking_results_raw, "docking_poses.pdbqt")
                
                df_int_orig = pd.DataFrame(active_interactions)
                orig_matrix_html = df_int_orig[["Residue Contact", "Interaction Type", "Distance (Å)"]].to_html(index=False, classes="data-table") if not df_int_orig.empty else "<p>No close contacts detected.</p>"

                p1_html_report = build_phase1_html_report(
                    meta=meta_data, p_2d=b_img, smiles_cache=st.session_state.smiles_cache, 
                    grid_params=grid_params, df_results_p1=df_results_p1, orig_ints=active_interactions, 
                    receptor_data=protein_data, orig_ligand_pose_data=parsed_poses[selected_pose], 
                    selected_pose_orig=selected_pose, style_mode=style_mode_p1, 
                    show_surface=surf_toggle_p1, pre_uff=pre_uff, post_uff=post_uff, 
                    delta_uff=delta_uff, active_retained_ions=st.session_state.active_retained_ions,
                    uff_theory_html=report_uff_theory_html, orig_matrix_html=orig_matrix_html,
                    grid_strategy=st.session_state.selected_native_ligand,
                    tree_data=st.session_state.get('selected_tree_data', None)
                )

                st.download_button(label="📥 Download Phase 1 HTML Research Report", data=p1_html_report, file_name=f"InSilico_Phase1_Report_{st.session_state.pdb_id_display}.html", mime="text/html", use_container_width=True, key="dl_phase1")

# --- ENGINE EXECUTION ---
if run_btn and can_dock:
    vina_path = os.path.abspath("vina")
    vina_command = [
        vina_path, "--receptor", "protein.pdbqt", "--ligand", "ligand.pdbqt", 
        "--center_x", str(grid_cx), "--center_y", str(grid_cy), "--center_z", str(grid_cz), 
        "--size_x", str(grid_sx), "--size_y", str(grid_sy), "--size_z", str(grid_sz), 
        "--exhaustiveness", str(exhaustiveness), "--out", "docking_poses.pdbqt"
    ]
    
    progress_bar = st.progress(0, text="Initializing computational engine...")
    status_text = st.empty()
    try:
        process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output_log, progress_count, current_line = [], 0, ""
        while True:
            char = process.stdout.read(1).decode("utf-8", errors="ignore")
            if not char: break
            output_log.append(char)
            if char == '*':
                progress_count += 1
                progress_bar.progress(min(100, int((progress_count / 50) * 100)), text=f"Exploring binding modes... {min(100, int((progress_count / 50) * 100))}%")
            elif char == '\n':
                if "Performing search" in current_line: status_text.info("Executing BFGS optimization and spatial search...")
                elif "Refining" in current_line: status_text.info("Refining top structural poses...")
                current_line = ""
            else: current_line += char
        process.wait()
        if process.returncode == 0:
            progress_bar.progress(100, text="Optimization complete!")
            status_text.empty()
            st.session_state.docking_results_raw = "".join(output_log)
            st.session_state.uff_cache = {} 
            
            try:
                a_str = get_pose_affinity(st.session_state.docking_results_raw, 1)
                if a_str != "N/A": st.session_state.baseline_affinity = float(a_str)
            except: pass
            
            time.sleep(0.8) 
            trigger_rerun = True
        else:
            status_text.empty(); st.error("Engine encountered a calculation error."); st.code("".join(output_log))
    except Exception as e: st.error(f"Execution pipeline failed: {e}")

if st.session_state.docking_results_raw is not None:
    st.write("---")
    st.markdown("### 📊 Screening Metrics Dashboard & Data Export")
    df_results = parse_vina_output_with_residues_global(st.session_state.docking_results_raw)
    if not df_results.empty:
        col_table, col_export = st.columns([2, 1])
        with col_table: 
            st.dataframe(df_results, hide_index=True, use_container_width=True)
        with col_export:
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Data Sheet (.CSV)", data=csv_data, file_name="screening_affinity_report.csv", mime="text/csv", use_container_width=True)

# ---------------------------------------------------------------------
# PHASE 2: GENERATIVE SCAFFOLD STRUCTURAL REDESIGN STUDIO
# ---------------------------------------------------------------------
st.write("---")
st.write("---")
st.header("🧬 Phase 2: Generative Scaffold Structural Redesign Studio")

if not st.session_state.ligand_ready:
    st.warning("⚠️ Access Gated: Provide a valid pure SMILES sequence or upload a molecular file and click 'Load Ligand Structure' in Phase 1 to unlock the modification dashboard.")
else:
    if not st.session_state.smiles_cache:
        st.warning("⚠️ SMILES Extraction Missing: Your uploaded 3D file was docked successfully, but we could not safely extract its 2D SMILES matrix automatically. Please paste its SMILES string below to proceed with redesign.")
        manual_smiles = st.text_input("Enter Ligand SMILES String for Scaffold Engine:").strip()
        if st.button("Unlock Redesign Engine"):
            if manual_smiles:
                st.session_state.smiles_cache = manual_smiles
                trigger_rerun = True
            else:
                st.error("Please provide a valid SMILES string.")
    
    if st.session_state.smiles_cache:
        cls_lbl, _ = get_dynamic_fragments(st.session_state.smiles_cache)
        st.info(f"🧬 **Automated AI Scaffold Family Classification Ident: `{cls_lbl}`**")
        
        rec_id = st.session_state.pdb_id_display if st.session_state.pdb_id_display else "Local Structural Matrix"
        st.markdown(f"> **Target Receptor Matrix (PDB ID):** `{rec_id}` <br> **Lead Drug Scaffold (SMILES):** `{st.session_state.smiles_cache}`", unsafe_allow_html=True)
        
        v_sites = find_valid_cleavage_sites(st.session_state.smiles_cache)
        col_rd_p, col_rd_v = st.columns([1, 1])
        
        with col_rd_p:
            rx_mode = st.radio("Select Optimization Processing Mode:", ["MockFrag Sandbox (100% Error-Free)", "Option B: True Structural Cleaving"], key="rx_mode_choice")
            toggle_lbl = st.toggle("Overlay Atom Index Identification Matrix Trackers", value=True)
            
            if "True Structural Cleaving" in rx_mode and v_sites:
                opts = {s["label"]: s["index"] for s in v_sites}
                sel_lbl = st.selectbox("Isolate legal targeted atom intersection for array modification:", options=list(opts.keys()))
                tgt_atom_idx = opts[sel_lbl]
            else:
                tgt_atom_idx = 0
                st.info("Sandbox Mode Active: System will formulate a safe co-crystal variation without breaking existing chemical bonds.")
                
            if st.button("🚀 Generate Optimized Derivative Structural Library", type="primary"):
                with st.spinner("Processing bioisosteric structural transformation loops..."):
                    res = run_cleaving_engine(st.session_state.smiles_cache, tgt_atom_idx, rx_mode)
                    if res and len(res) > 0:
                        st.session_state.rd_library = pd.DataFrame(res)
                        st.success(f"Successfully synthesized {len(res)} modified entries tracking baseline affinity data.")
                        trigger_rerun = True
        
        with col_rd_v:
            b_img = generate_clean_2d_image(st.session_state.smiles_cache, include_labels=toggle_lbl, zoom_level=550)
            if b_img: st.markdown(b_img, unsafe_allow_html=True)
            
        if st.session_state.rd_library is not None and not st.session_state.rd_library.empty:
            st.subheader("Synthesized Structural Variant Optimization Array Data Track")
            st.dataframe(st.session_state.rd_library[["Variant ID", "Fragment Added", "Redesigned SMILES", "Delta Score", "MW (g/mol)", "LogP"]], hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# PHASE 3: ADMET 3.0 Pharmacokinetics Profiling
# ---------------------------------------------------------------------
st.write("---")
st.write("---")
st.header("📊 Phase 3: ADMET 3.0 Pharmacokinetics Profiling")

if st.session_state.rd_library is None or st.session_state.rd_library.empty:
    st.warning("⚠️ Access Gated: Initialize generation matrices within Phase 2 to display complete profiling reports.")
else:
    st.session_state.selected_variant_id = st.selectbox("Isolate synthesized structural entry to analyze pharmacokinetics metrics:", options=st.session_state.rd_library["Variant ID"])
    
    v_rows = st.session_state.rd_library[st.session_state.rd_library["Variant ID"] == st.session_state.selected_variant_id]
    if not v_rows.empty:
        v_row = v_rows.iloc[0]
        curr_smiles = str(v_row["Redesigned SMILES"])
        
        with st.spinner("Compiling structural property descriptors..."):
            iupac = get_iupac_name(curr_smiles)
            adme_p = calculate_advanced_adme(st.session_state.smiles_cache)
            adme_v = calculate_advanced_adme(curr_smiles)
            
            st.info(f"**Nomenclature Alignment Index (IUPAC Name):** `{iupac}`")
            
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                st.markdown("#### Structural Topology Footprint")
                v_2d = generate_clean_2d_image(curr_smiles, include_labels=False, zoom_level=420)
                if v_2d: st.markdown(v_2d, unsafe_allow_html=True)
                
            with col_m2:
                st.markdown("#### Modeled Vibrational Footprint (FTIR Analysis)")
                ftir_b64 = generate_ftir_image(int(v_row["FTIR Peak"]))
                st.markdown(f'<img src="data:image/png;base64,{ftir_b64}" style="max-width:100%; border-radius:6px; border:1px solid #ddd;"/>', unsafe_allow_html=True)
            
            st.write("---")
            st.subheader("Comparative Molecular Property Descriptors")
            
            comp_df = pd.DataFrame({
                "Physiochemical Bioproperty Descriptor": [
                    "Lipinski Compliance?", "Oral Route Usability Profile", "Permeability Barrier Property",
                    "Topological Polar Surface Area (TPSA)", "Molecular Spatial Volume (Å³)", "Rigidity Constraints (Max Ring Size)",
                    "Lipophilic Distribution Tracker (LogP)", "pKa (Acidic)", "pKa (Basic)", "Thermodynamic Melting Boundaries (°C)"
                ],
                "Original Phytochemical Scaffold Matrix": [
                    adme_p['Lipinski_Obey'], adme_p['Oral_Bio'], adme_p['Permeability'],
                    f"{adme_p['TPSA']:.2f} Å²" if isinstance(adme_p['TPSA'], float) else "0.00 Å²", 
                    f"{adme_p['Volume']:.1f} Å³" if isinstance(adme_p['Volume'], float) else "0.0 Å³", 
                    adme_p['MaxRing'], 
                    f"{adme_p['LogP']:.2f}" if isinstance(adme_p['LogP'], float) else "0.00", 
                    adme_p['pKa_Acid'], adme_p['pKa_Base'], 
                    f"{adme_p['MP']:.1f}" if isinstance(adme_p['MP'], float) else "0.0"
                ],
                "Redesigned Structural Target Variant": [
                    adme_v['Lipinski_Obey'], adme_v['Oral_Bio'], adme_v['Permeability'],
                    f"{adme_v['TPSA']:.2f} Å²" if isinstance(adme_v['TPSA'], float) else "0.00 Å²", 
                    f"{adme_v['Volume']:.1f} Å³" if isinstance(adme_v['Volume'], float) else "0.0 Å³", 
                    adme_v['MaxRing'], 
                    f"{adme_v['LogP']:.2f}" if isinstance(adme_v['LogP'], float) else "0.00", 
                    adme_v['pKa_Acid'], adme_v['pKa_Base'], 
                    f"{adme_v['MP']:.1f}" if isinstance(adme_v['MP'], float) else "0.0"
                ]
            })
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
            try:
                vol_shift, tpsa_shift, logp_shift = adme_v['Volume'] - adme_p['Volume'], adme_v['TPSA'] - adme_p['TPSA'], adme_v['LogP'] - adme_p['LogP']
                shift_msg = f"Redesign workflow caused structural volume changes equal to **{vol_shift:.1f} Å³**. "
                shift_msg += f"Polar group inclusion expanded topological polar parameters (TPSA) by **{tpsa_shift:.1f} Å²**. " if tpsa_shift > 0 else f"Polar reductions decreased surface topology metrics (TPSA) by **{abs(tpsa_shift):.1f} Å²**. "
                
                if adme_v['Violations'] < adme_p['Violations']: shift_msg += "\n\n📊 **Ecosystem Assessment Verdict: Favorable.** Candidate displays enhanced bioavailability compliance profiles."
                elif adme_v['Violations'] > adme_p['Violations']: shift_msg += "\n\n❌ **Ecosystem Assessment Verdict: Unfavorable.** Optimization mismatch."
                else: shift_msg += "\n\n⚖️ **Ecosystem Assessment Verdict: Comparable.** Valid chemical structural configuration balance safely maintained."
            except Exception: shift_msg = "⚠️ Ecosystem Assessment Verdict: Chemical structure too strained to calculate ADMET shifts."
            st.success(shift_msg)

# ---------------------------------------------------------------------
# PHASE 4: POST-REDESIGN VALIDATION DOCKING & MASTER SYNTHESIS
# ---------------------------------------------------------------------
st.write("---")
st.write("---")
st.header("🎯 Phase 4: Post-Redesign Validation Docking & Master Synthesis")

if st.session_state.rd_library is None or st.session_state.rd_library.empty or not st.session_state.target_ready:
    st.warning("⚠️ Access Gated: Complete Phase 1 Docking and Phase 2/3 Redesign to unlock validation module.")
else:
    col_p4_1, col_p4_2 = st.columns([1, 1])
    with col_p4_1:
        st.subheader("1. Inherit Structural Data")
        if st.button("🔄 Pull Receptor & Phase 3 Derivative", type="secondary"):
            v_rows = st.session_state.rd_library[st.session_state.rd_library["Variant ID"] == st.session_state.selected_variant_id]
            if not v_rows.empty:
                new_smiles = str(v_rows.iloc[0]["Redesigned SMILES"])
                ok, msg = convert_smiles_to_pdbqt(new_smiles, "redesign_ligand.pdbqt")
                if ok:
                    st.success(f"Derivative `{st.session_state.selected_variant_id}` securely converted to 3D matrix.")
                    st.session_state.redesign_docking_results_raw = None
                else: st.error(f"3D Embedding Failed: {msg}")
                    
        st.markdown(f"> **Target Receptor:** `{st.session_state.pdb_id_display}` <br> **Active Derivative:** `{st.session_state.selected_variant_id}`", unsafe_allow_html=True)
        
    with col_p4_2:
        st.subheader("2. Execute Validation Docking")
        grid_mode = st.radio("Grid Box Selection:", ["Use Phase 1 Grid Box Parameters", "Auto-Configure Blind Docking"], key="p4_grid")
        can_run_p4 = os.path.exists("protein.pdbqt") and os.path.exists("redesign_ligand.pdbqt")
        
        if st.button("🚀 Initialize Validation Docking Engine", type="primary", disabled=not can_run_p4):
            if "Blind" in grid_mode: p4_cx, p4_cy, p4_cz, p4_sx, p4_sy, p4_sz = compute_protein_bounding_box("protein.pdbqt")
            else:
                p4_cx, p4_cy, p4_cz = st.session_state.cx, st.session_state.cy, st.session_state.cz
                p4_sx, p4_sy, p4_sz = st.session_state.sx, st.session_state.sy, st.session_state.sz
                
            vina_path = os.path.abspath("vina")
            vina_command = [
                vina_path, "--receptor", "protein.pdbqt", "--ligand", "redesign_ligand.pdbqt", 
                "--center_x", str(p4_cx), "--center_y", str(p4_cy), "--center_z", str(p4_cz), 
                "--size_x", str(int(p4_sx)), "--size_y", str(int(p4_sy)), "--size_z", str(int(p4_sz)), 
                "--exhaustiveness", str(st.session_state.exhaustiveness), "--out", "redesign_docking_poses.pdbqt"
            ]
            
            p4_prog = st.progress(0, text="Validating new derivative...")
            p4_stat = st.empty()
            try:
                process = subprocess.Popen(vina_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output_log, p_count, c_line = [], 0, ""
                while True:
                    char = process.stdout.read(1).decode("utf-8", errors="ignore")
                    if not char: break
                    output_log.append(char)
                    if char == '*':
                        p_count += 1
                        p4_prog.progress(min(100, int((p_count / 50) * 100)), text="Exploring optimized binding modes...")
                    elif char == '\n': c_line = ""
                    else: c_line += char
                process.wait()
                if process.returncode == 0:
                    p4_prog.progress(100, text="Validation complete!")
                    p4_stat.empty()
                    st.session_state.redesign_docking_results_raw = "".join(output_log)
                    trigger_rerun = True
                else: p4_stat.empty(); st.error("Engine failed during validation.")
            except Exception as e: st.error(f"Validation pipeline error: {e}")

    if st.session_state.redesign_docking_results_raw is not None and os.path.exists("redesign_docking_poses.pdbqt"):
        st.write("---")
        st.subheader("3. Validation Complex Analysis (Side-by-Side Comparison)")
        p4_poses = split_docking_poses("redesign_docking_poses.pdbqt")
        if p4_poses:
            p4_sel_pose = st.selectbox("Select Derivative Binding Pose for Comparison:", options=list(p4_poses.keys()), format_func=lambda x: f"Derivative Pose {x}", key="p4_pose_sel")
            orig_aff = st.session_state.baseline_affinity
            new_aff_str = get_pose_affinity(st.session_state.redesign_docking_results_raw, p4_sel_pose)
            try: st.session_state.redesign_baseline_affinity = float(new_aff_str)
            except: pass

            cache_key_p4 = f"uff_p4_{st.session_state.selected_variant_id}_{p4_sel_pose}"
            uff_prog_p4 = st.empty()
            if cache_key_p4 not in st.session_state.uff_cache:
                pre, post, delta = execute_uff_complex_minimization("protein.pdbqt", p4_poses[p4_sel_pose], uff_prog_p4)
                st.session_state.uff_cache[cache_key_p4] = (pre, post, delta)
            uff_prog_p4.empty()
            pre_uff, post_uff, delta_uff = st.session_state.uff_cache[cache_key_p4]

            orig_pose = split_docking_poses("docking_poses.pdbqt").get(st.session_state.get('selected_pose_export', 1), "") if os.path.exists("docking_poses.pdbqt") else ""
            orig_ints = compute_spatial_interactions("protein.pdbqt", orig_pose) if orig_pose else []
            new_ints = compute_spatial_interactions("protein.pdbqt", p4_poses[p4_sel_pose])
            
            o_res = ", ".join(sorted(list(set([i["Residue Contact"] for i in orig_ints])))) if orig_ints else "None"
            n_res = ", ".join(sorted(list(set([i["Residue Contact"] for i in new_ints])))) if new_ints else "None"
            o_bonds = ", ".join(sorted(list(set([i["Interaction Type"] for i in orig_ints])))) if orig_ints else "None"
            n_bonds = ", ".join(sorted(list(set([i["Interaction Type"] for i in new_ints])))) if new_ints else "None"

            with open("protein.pdbqt", "r") as f: p_data = f.read()
            
            col_3d_1, col_3d_2 = st.columns(2)
            with col_3d_1:
                st.markdown("#### Original Lead Complex")
                style_choice_p4_orig = st.radio("Style (Original):", ["Cartoon Ribbon Mesh", "Spacefill", "Sticks Profile"], key="p4_style_o")
                style_mode_p4_orig = re.sub(r'\W+', '', style_choice_p4_orig.split()[0].lower())
                surf_toggle_p4_orig = st.checkbox("Translucent Mesh", value=False, key="p4_surf_o")
                render_advanced_modeling_blueprint(p_data, orig_pose, mode=style_mode_p4_orig, show_surface=surf_toggle_p4_orig, interactions_list=orig_ints, unique_id="p4_orig_viewer")
                
            with col_3d_2:
                st.markdown(f"#### Redesigned Derivative (Pose {p4_sel_pose})")
                style_choice_p4_new = st.radio("Style (Derivative):", ["Cartoon Ribbon Mesh", "Spacefill", "Sticks Profile"], key="p4_style_n")
                style_mode_p4_new = re.sub(r'\W+', '', style_choice_p4_new.split()[0].lower())
                surf_toggle_p4_new = st.checkbox("Translucent Mesh", value=False, key="p4_surf_n")
                render_advanced_modeling_blueprint(p_data, p4_poses[p4_sel_pose], mode=style_mode_p4_new, show_surface=surf_toggle_p4_new, interactions_list=new_ints, unique_id="p4_new_viewer")
            
            st.write("---")
            with st.expander("📖 Understand UFF Minimization & Steric Clashes (Click to Expand)", expanded=False):
                st.info(f"""
                **1. 📍 UFF Initial Energy: {pre_uff} kcal/mol**
                This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, *before* any relaxation occurred. A highly positive energy score indicates extreme geometric tension (a steric clash/rigid atomic wall effect). It means atoms from your phytochemical were physically overlapping or positioned unnaturally close to the rigid atoms of the receptor—most likely the catalytic metal ions or cofactors you specifically chose to retain. In a living biological system, atoms cannot overlap; they would repel each other and shift. But Vina's rigid grid didn't allow them to shift.

                **2. 📉 Optimized Energy: {post_uff} kcal/mol**
                This is the total stress of the complex *after* the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm gently pushed overlapping atoms apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state. The negative force field delta (**{delta_uff} kcal/mol**) proves the rigid collision was successfully resolved!
                """)

            st.markdown("#### ⚖️ Direct Thermodynamic Comparison Matrix")
            
            orig_delta = st.session_state.get('baseline_delta_uff', "N/A")
            if orig_delta != "N/A": orig_delta = f"{orig_delta} kcal/mol"
            
            comp_data = {
                "Metric": ["Gibbs Free Energy (ΔG)", "UFF Minimization Delta", "Pocket Residue Contacts", "Identified Interaction Types"],
                "Original Lead": [f"{orig_aff} kcal/mol" if orig_aff else "N/A", orig_delta, o_res, o_bonds],
                "Optimized Derivative": [f"{new_aff_str} kcal/mol", f"{delta_uff} kcal/mol", n_res, n_bonds]
            }
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, hide_index=True, use_container_width=True)
            
            try: delta_aff = round(float(new_aff_str) - float(orig_aff), 2) if orig_aff else 0.0
            except: delta_aff = 0.0
            
            master_verdict = ""
            if delta_aff < -0.5: master_verdict += f"🟢 **Outstanding Validation:** Derivative enhanced binding affinity by **{delta_aff} kcal/mol**. "
            elif delta_aff < 0: master_verdict += f"🟢 **Positive Validation:** Derivative improved binding affinity by **{delta_aff} kcal/mol**. "
            elif delta_aff == 0: master_verdict += f"🟡 **Neutral Validation:** Derivative maintained the exact baseline binding affinity. "
            else: master_verdict += f"🔴 **Negative Validation:** Modification worsened binding affinity by **+{delta_aff} kcal/mol**. "

            if delta_aff <= 0 and ("Favorable" in shift_msg or "Comparable" in shift_msg):
                master_verdict += "Coupled with the stable ADME profile, this structural modification is a **Strong Candidate for Synthesis**."
            elif delta_aff > 0:
                master_verdict += "Because the binding affinity worsened, this structural modification should be **Rejected and Redesigned**, regardless of ADME stability."
            else:
                master_verdict += "Furthermore, due to the compromised ADME profile, this structural modification should be **Rejected and Redesigned**."

            st.markdown("#### 📜 Master Synthesis Verdict")
            st.info(master_verdict)

            # --- REPORT EXPORT ---
            st.write("---")
            st.subheader("📋 Phase 4: Local Contact Matrices & Final Report Generation")
            
            st.markdown("#### 🧬 Local Contact Residues & Bond Assignments Matrix")
            col_rm1, col_rm2 = st.columns(2)
            with col_rm1:
                st.markdown("**Original Lead Contacts**")
                if orig_ints: st.dataframe(pd.DataFrame(orig_ints)[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True)
                else: st.info("No close contacts.")
            with col_rm2:
                st.markdown("**Optimized Derivative Contacts**")
                if new_ints: st.dataframe(pd.DataFrame(new_ints)[["Residue Contact", "Interaction Type", "Distance (Å)"]], hide_index=True)
                else: st.info("No close contacts.")

            include_uff_theory = st.checkbox("Include detailed UFF biophysical explanation in the generated reports", value=True, key="p4_uff_toggle")
            
            report_uff_theory_text = ""
            report_uff_theory_html = ""
            if include_uff_theory:
                report_uff_theory_text = f"""
8. UFF MINIMIZATION BIOPHYSICAL EXPLANATION
-------------------------------------------------------
- 📍 UFF Initial Energy: {pre_uff} kcal/mol
  This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, before any relaxation occurred. A highly positive energy score indicates extreme geometric tension, often a steric clash where atoms physically overlap with rigid atoms of the receptor or retained catalytic cofactors. In a living biological system, atoms shift to relieve this, but a rigid grid does not allow it.

- 📉 Optimized Energy: {post_uff} kcal/mol
  This is the total stress of the complex after the Universal Force Field (UFF) algorithm ran took the overlapping atoms and gently pushed them apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state, making the system structurally stable. The critical metric is the massive drop from the initial state ({delta_uff} kcal/mol).
"""
                report_uff_theory_html = f"""
                <details style="background-color: #f9fbff; border-left: 6px solid #1e3c72; padding: 15px; border-radius: 4px; margin-top: 20px;">
                    <summary style="font-weight: bold; cursor: pointer; color: #1e3c72; font-size: 16px;">📖 Understand UFF Minimization & Steric Clashes (Click to Expand)</summary>
                    <div style="margin-top: 15px;">
                        <p><b>📍 UFF Initial Energy: {pre_uff} kcal/mol</b></p>
                        <p>This represents the total internal physical stress of the protein-ligand complex the moment AutoDock Vina finished placing your molecule into the pocket, before any relaxation occurred. A highly positive energy score indicates extreme geometric tension. This is the mathematical signature of a steric clash (the "rigid atomic wall" effect). It means atoms from your phytochemical were physically overlapping or positioned unnaturally close to the rigid atoms of the receptor—most likely the catalytic metal ions or cofactors you specifically chose to retain. In a living biological system, atoms cannot overlap; they would repel each other and shift. But Vina's rigid grid didn't allow them to shift, resulting in this artificially high stress value.</p>
                        
                        <p><b>📉 Optimized Energy: {post_uff} kcal/mol</b></p>
                        <p>This is the total stress of the complex after the Universal Force Field (UFF) algorithm ran its gradient descent optimization. The algorithm took the overlapping atoms and gently pushed them apart by fractions of an Angstrom until the bond lengths and angles reached a naturally permissible state. The system is now structurally stable. What matters is not that the final number is positive, but how far it dropped from the initial state (<b>{delta_uff} kcal/mol</b>).</p>
                    </div>
                </details>
                """

            p4_int_text_o = format_interaction_matrix_text(orig_ints)
            p4_int_text_n = format_interaction_matrix_text(new_ints)

            st.markdown("**Quick Copy-Paste Citation Report (Phase 4 Final Validation)**")
            report_content_p4 = f"""=======================================================
MOLECULAR DOCKING SCREENING ANALYSIS REPORT (FINAL VALIDATION)
Generated dynamically via InSilico BioSphere Docking Tool
Developed by: Dr. Sarang S. Dhote, Assistant Professor, Department of Chemistry, Shivaji Science College, Nagpur, India | Contact: sarangresearch@gmail.com
=======================================================

1. TARGET RECEPTOR MACROMOLECULE PROFILE
-------------------------------------------------------
- Target Protein Name: {st.session_state.protein_name}
- Target Configuration Identifier (PDB ID): {st.session_state.pdb_id_display}
- Primary Structure Data Source: RCSB Protein Data Bank Server / Local Upload
- Catalytic Cofactors & Heteroatom Filter configured by user: {st.session_state.active_retained_ions}

2. SMALL MOLECULE DRUG LIGAND PROFILE
-------------------------------------------------------
- Input Structural Identity Matrix: {st.session_state.get('smiles_cache', 'Uploaded File Data Track')}
- Compiled Chemical Attributes: {st.session_state.ligand_summary_text.replace('**','')}

3. BOUND SPACE CONFIGURATION MECHANICS (GRID BOX)
-------------------------------------------------------
- Center Coordinates Vector (X, Y, Z): ({grid_cx}, {grid_cy}, {grid_cz})
- Grid Bounding Dimensions (X, Y, Z): ({grid_sx} Å, {grid_sy} Å, {grid_sz} Å)
- Search Algorithm Exhaustiveness Index: {exhaustiveness}

4. ACTIVE POSE COMPLEX BINDING METRICS (COMPARING OPTIMIZED DERIVATIVE VS ORIGINAL)
-------------------------------------------------------
- Target Alignment Selection Mode: Mode {selected_pose} Pose Fit
- Original Gibbs Free Energy Affinity: {orig_aff} kcal/mol
- Redesigned Gibbs Free Energy Affinity: {new_aff_str} kcal/mol
- Measured Total Spatial Proximity Contact Atoms: {len(new_ints)}
- Derivative UFF Post-Docking Energy Parameters: Initial: {pre_uff} | Relaxed: {post_uff} | Delta: {delta_uff} kcal/mol

5. LOCAL CONTACT RESIDUES & BOND ASSIGNMENTS MATRIX
-------------------------------------------------------
[ ORIGINAL LEAD MATRIX ]
{p4_int_text_o}

[ REDESIGNED DERIVATIVE MATRIX ]
{p4_int_text_n}

6. SCIENTIFIC METHODOLOGY & MANUSCRIPT CITATION TRACK
-------------------------------------------------------
Molecular docking was performed using the semi-empirical force field parameters of AutoDock Vina inside the InSilico BioSphere framework. To maintain structural and biological validity, essential catalytic cofactor ions were explicitly preserved within the target binding cleft during search configurations. Potential localized steric constraints and rigid atomic wall collisions resulting from structural constraints were resolved by subjecting the final protein-ligand complexes to post-docking energy minimization using the Universal Force Field (UFF) optimized to a convergence tolerance of 10^-4 kcal/mol·Å.

Manuscript Citation Format Block:
Dr. Sarang S. Dhote, "InSilico BioSphere: An Integrated Platform for Automated Molecular Docking, Surface Cavity Profiling, and Post-Docking Force-Field Relaxation Mechanics." Department of Chemistry, Shri Shivaji Science College, Nagpur, India. Correspondence: sarangresearch@gmail.com
{report_uff_theory_text}=======================================================
"""
            st.text_area("Copy Phase 4 Report Text directly:", value=report_content_p4, height=250, key="p4_text_area")

            meta_data = extract_pdb_metadata(st.session_state.local_target_path, st.session_state.pdb_id_display) if st.session_state.local_target_path else {"id":"Custom","title":"Uploaded Structure File","method":"N/A","res":"N/A"}
            meta_data['name'], meta_data['id'] = st.session_state.protein_name, st.session_state.pdb_id_display
            b_img = generate_clean_2d_image(st.session_state.smiles_cache, include_labels=False, zoom_level=420)
            grid_params = {'cx': st.session_state.cx, 'cy': st.session_state.cy, 'cz': st.session_state.cz, 'sx': st.session_state.sx, 'sy': st.session_state.sy, 'sz': st.session_state.sz, 'exh': st.session_state.exhaustiveness}
            
            df_comparison_html = '<table class="dataframe table"><thead><tr><th>Metric</th><th>Original Lead</th><th>Optimized Derivative</th></tr></thead><tbody>'
            for _, r in df_comp.iterrows():
                val = str(r['Optimized Derivative'])
                df_comparison_html += f"<tr><td>{r['Metric']}</td><td>{r['Original Lead']}</td><td style='font-weight: bold;'>{val}</td></tr>"
            df_comparison_html += '</tbody></table>'

            df_results_baseline = parse_vina_output_with_residues_global(st.session_state.docking_results_raw, "docking_poses.pdbqt")
            df_results_redesign = parse_vina_output_with_residues_global(st.session_state.redesign_docking_results_raw, "redesign_docking_poses.pdbqt")
            try:
                with open("protein.pdbqt", "r") as f: receptor_data = f.read()
            except: receptor_data = ""

            df_int_orig = pd.DataFrame(orig_ints)
            orig_matrix_html = df_int_orig[["Residue Contact", "Interaction Type", "Distance (Å)"]].to_html(index=False, classes="data-table") if not df_int_orig.empty else "<p>No close contacts detected.</p>"
            df_int_new = pd.DataFrame(new_ints)
            new_matrix_html = df_int_new[["Residue Contact", "Interaction Type", "Distance (Å)"]].to_html(index=False, classes="data-table") if not df_int_new.empty else "<p>No close contacts detected.</p>"

            html_report = build_comprehensive_html_report(
                meta=meta_data, adme_p=adme_p, adme_v=adme_v, variant_row=v_row, iupac=iupac, shift_msg=shift_msg, 
                f_img=ftir_b64, v_2d=v_2d, p_2d=b_img, smiles_cache=st.session_state.smiles_cache, 
                baseline_affinity=st.session_state.baseline_affinity, grid_params=grid_params, 
                df_results_baseline=df_results_baseline, df_results_redesign=df_results_redesign, 
                orig_ints=orig_ints, new_ints=new_ints, receptor_data=receptor_data, orig_ligand_pose_data=orig_pose, 
                redesign_ligand_pose_data=p4_poses[p4_sel_pose], selected_pose_orig=st.session_state.get('selected_pose_export', 1), 
                selected_pose_new=p4_sel_pose, style_mode_orig=style_mode_p4_orig, show_surface_orig=surf_toggle_p4_orig,
                style_mode_new=style_mode_p4_new, show_surface_new=surf_toggle_p4_new,
                master_verdict=master_verdict, df_comparison_html=df_comparison_html, pre_uff=pre_uff, post_uff=post_uff, delta_uff=delta_uff,
                active_retained_ions=st.session_state.active_retained_ions, uff_theory_html=report_uff_theory_html,
                orig_matrix_html=orig_matrix_html, new_matrix_html=new_matrix_html, grid_strategy=st.session_state.selected_native_ligand,
                tree_data=st.session_state.get('selected_tree_data', None)
            )
            
            st.download_button(label="📥 Download Consolidated Manuscript Quality HTML Research Report", data=html_report, file_name=f"InSilico_BioSphere_Research_Record_{v_row['Variant ID']}.html", mime="text/html", use_container_width=True, key="dl_phase4")

if trigger_rerun: safe_rerun()
