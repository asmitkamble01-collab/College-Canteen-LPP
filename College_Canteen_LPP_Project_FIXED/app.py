
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Canteen Cost Minimization", layout="wide")

# ------------------------------------------------------------
# Sample canteen dataset: cost (Rs) and nutrition per serving
# Values are illustrative project data and can be replaced by
# actual canteen measurements/prices.
# ------------------------------------------------------------
FOODS = pd.DataFrame([
    ["Poha",       25,  5.4, 273, 43.6,  8.5],
    ["frankie",       50,  4.9, 339, 40,  18],
    ["Upma",       30,  6, 210, 38,  4],
    ["Misal Pav",  50, 14, 450, 60, 18],
    ["Pav bhaji",50, 9.4, 478, 54.6, 24.7],
    ["Dal Rice",   50, 10, 320, 50, 8],
    ["dabeli",20, 4.2, 219, 31.6, 8.3],
    ["samosa pav",    18, 6, 298, 36, 15],
], columns=["Food", "Cost", "Protein", "Calories", "Carbs", "Fat"])


# ------------------------------------------------------------
# General two-phase simplex for minimization with >= constraints
#
# Input:
#   c      = objective coefficients (minimize c^T x)
#   A      = constraint matrix
#   b      = RHS
#   senses = >=, <=, or =
#
# The implementation converts constraints to a tableau and uses
# Phase I artificial variables to obtain a feasible basis.
# Phase II minimizes the original objective.
# ------------------------------------------------------------
def build_tableau(c, A, b, senses):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Make RHS non-negative.
    rows = []
    senses2 = []
    for i in range(len(b)):
        row = A[i].copy()
        rhs = b[i]
        sense = senses[i]
        if rhs < 0:
            row = -row
            rhs = -rhs
            if sense == ">=":
                sense = "<="
            elif sense == "<=":
                sense = ">="
        rows.append(row)
        senses2.append(sense)

    A = np.array(rows, dtype=float)
    b = np.array([b_i if b_i >= 0 else -b_i for b_i in b], dtype=float)

    n = len(c)
    cols = [f"x{i+1}" for i in range(n)]
    matrix_cols = [A[:, j].copy() for j in range(n)]
    artificial = []
    basis = []

    for i, sense in enumerate(senses2):
        if sense == "<=":
            col = np.zeros(len(b))
            col[i] = 1.0
            matrix_cols.append(col)
            cols.append(f"s{i+1}")
            basis.append(len(cols) - 1)
        elif sense == ">=":
            col = np.zeros(len(b))
            col[i] = -1.0
            matrix_cols.append(col)
            cols.append(f"e{i+1}")  # surplus
            # artificial variable
            acol = np.zeros(len(b))
            acol[i] = 1.0
            matrix_cols.append(acol)
            cols.append(f"A{i+1}")
            artificial.append(len(cols) - 1)
            basis.append(len(cols) - 1)
        else:  # equality
            acol = np.zeros(len(b))
            acol[i] = 1.0
            matrix_cols.append(acol)
            cols.append(f"A{i+1}")
            artificial.append(len(cols) - 1)
            basis.append(len(cols) - 1)

    M = np.column_stack(matrix_cols)
    return M, b, cols, basis, artificial, c


def canonical_tableau(M, b, basis, cj):
    """Return a Cj-Zj tableau for a maximization-style objective."""
    m, n = M.shape
    cb = np.array([cj[j] for j in basis], dtype=float)
    zj = cb @ M
    z0 = cb @ b
    delta = cj - zj

    tab = np.column_stack([b, M])
    return tab, cb, zj, z0, delta


def pivot(M, b, basis, row, col):
    M = M.copy()
    b = b.copy()

    pivot_value = M[row, col]
    M[row, :] /= pivot_value
    b[row] /= pivot_value

    for i in range(len(b)):
        if i == row:
            continue
        factor = M[i, col]
        if abs(factor) > 1e-12:
            M[i, :] -= factor * M[row, :]
            b[i] -= factor * b[row]

    basis = basis.copy()
    basis[row] = col
    return M, b, basis


def run_phase(M, b, basis, objective, names, phase_name, max_iter=100):
    """Maximize objective using Cj-Zj convention."""
    M = M.copy()
    b = b.copy()
    basis = basis.copy()
    objective = np.array(objective, dtype=float)

    snapshots = []
    m, n = M.shape

    for iteration in range(max_iter):
        tab, cb, zj, z0, delta = canonical_tableau(M, b, basis, objective)

        snapshots.append({
            "iteration": iteration,
            "phase": phase_name,
            "basis": [names[j] for j in basis],
            "tableau": tab.copy(),
            "cb": cb.copy(),
            "delta": delta.copy(),
            "z": z0,
        })

        # For maximization, optimal when all Cj-Zj <= 0.
        if np.max(delta) <= 1e-9:
            return M, b, basis, snapshots, True, "Optimal"

        entering_candidates = np.where(delta > 1e-9)[0]
        entering = entering_candidates[np.argmax(delta[entering_candidates])]

        ratios = []
        for i in range(m):
            if M[i, entering] > 1e-9:
                ratios.append((b[i] / M[i, entering], i))
        if not ratios:
            return M, b, basis, snapshots, False, "Unbounded"

        _, leaving = min(ratios, key=lambda x: x[0])
        M, b, basis = pivot(M, b, basis, leaving, entering)

    return M, b, basis, snapshots, False, "Iteration limit reached"


def remove_artificial_columns(M, names, basis, artificial):
    keep = [j for j in range(len(names)) if j not in artificial]
    mapping = {old: new for new, old in enumerate(keep)}
    new_basis = []

    # If an artificial variable remains basic at value 0, try to pivot
    # a non-artificial column into that row.
    M2 = M[:, keep].copy()
    basis2 = basis.copy()

    for i, old_basic in enumerate(basis):
        if old_basic in artificial:
            candidates = [j for j in range(M2.shape[1])
                          if abs(M2[i, j]) > 1e-9 and j not in basis2]
            if candidates:
                # Find a candidate that preserves a valid pivot.
                M2, dummy_b, basis2 = pivot(M2, np.zeros(len(basis2)), basis2,
                                             i, candidates[0])
            else:
                # Redundant zero row can be removed by caller if needed.
                pass

    for old in basis2:
        if old in artificial:
            # This can only happen for a redundant zero row.
            new_basis.append(None)
        else:
            new_basis.append(mapping[old])

    return M2, [names[j] for j in keep], new_basis


def solve_lpp(c, A, b, senses):
    M, rhs, names, basis, artificial, original_c = build_tableau(c, A, b, senses)

    # Phase I: maximize -sum artificial variables.
    phase1_obj = np.zeros(len(names))
    for j in artificial:
        phase1_obj[j] = -1.0

    M1, b1, basis1, phase1_snaps, ok1, status1 = run_phase(
        M, rhs, basis, phase1_obj, names, "Phase I"
    )

    # Check Phase I objective. It must be 0.
    _, _, _, z1, _ = canonical_tableau(M1, b1, basis1, phase1_obj)
    if not ok1 or z1 < -1e-7:
        return {"status": "Infeasible", "phase1": phase1_snaps}

    # Remove artificial variables and convert the Phase-I basis
    # from old-column indices to reduced-column indices.
    keep = [j for j in range(len(names)) if j not in artificial]
    reduced_names = [names[j] for j in keep]
    old_to_new = {old: new for new, old in enumerate(keep)}
    M2 = M1[:, keep].copy()
    b2 = b1.copy()

    basis2 = [old_to_new.get(j, -1) for j in basis1]

    # If an artificial variable is still basic at zero, replace it with a
    # non-artificial column whenever possible.
    for i, bj in enumerate(basis2):
        if bj == -1:
            candidates = [
                j for j in range(M2.shape[1])
                if abs(M2[i, j]) > 1e-9 and j not in basis2
            ]
            if candidates:
                M2, b2, basis2 = pivot(
                    M2, b2, basis2, i, candidates[0]
                )
            else:
                # A zero artificial basic row is redundant. For the
                # nutrition model used here this case should not occur.
                return {"status": "Infeasible", "phase1": phase1_snaps}

    b1 = b2

    # Phase II: minimize c^T x = maximize -c^T x.
    phase2_obj = np.zeros(len(reduced_names))
    # The first columns are the original decision variables x1...xn.
    # Slack/surplus columns are not part of original_c.
    phase2_obj[:len(original_c)] = -original_c

    M_final, b_final, basis_final, phase2_snaps, ok2, status2 = run_phase(
        M2, b1, basis2, phase2_obj, reduced_names, "Phase II"
    )

    if not ok2:
        return {
            "status": status2,
            "phase1": phase1_snaps,
            "phase2": phase2_snaps,
        }

    x = np.zeros(len(reduced_names))
    for i, bj in enumerate(basis_final):
        x[bj] = b_final[i]

    # Extract only original decision variables.
    decision = x[:len(original_c)]
    cost = float(np.dot(original_c, decision))

    return {
        "status": "Optimal",
        "x": decision,
        "cost": cost,
        "names": reduced_names,
        "phase1": phase1_snaps,
        "phase2": phase2_snaps,
        "A": np.array(A, dtype=float),
        "b": np.array(b, dtype=float),
        "senses": senses,
    }


def display_snapshots(snapshots, names):
    for snap in snapshots:
        st.markdown(f"**{snap['phase']} — Iteration {snap['iteration']}**")
        tab = snap["tableau"]
        cols = ["b"] + names
        df = pd.DataFrame(tab, columns=cols)
        df.index = [f"R{i+1}" for i in range(len(df))]
        st.dataframe(df.round(4), use_container_width=True)
        delta = pd.DataFrame([snap["delta"]], columns=names)
        delta.index = ["Cj - Zj"]
        st.dataframe(delta.round(4), use_container_width=True)


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("🍱 College Canteen Cost Minimization & Diet Optimization")
st.caption("Linear Programming Problem solved using the Two-Phase Simplex Method")

st.info(
    "Project model: choose quantities of canteen items so that minimum daily "
    "nutrition requirements are satisfied at minimum cost."
)

st.subheader("1. Canteen Food Data")
edited = st.data_editor(
    FOODS,
    use_container_width=True,
    num_rows="fixed",
    disabled=["Food"],
)

st.subheader("2. Daily Minimum Nutrition Requirements")
c1, c2, c3, c4 = st.columns(4)
protein_req = c1.number_input("Protein (g) minimum", min_value=0.0, value=60.0, step=5.0)
cal_req = c2.number_input("Calories minimum", min_value=0.0, value=1800.0, step=100.0)
carb_req = c3.number_input("Carbs (g) minimum", min_value=0.0, value=250.0, step=10.0)
fat_req = c4.number_input("Fat (g) minimum", min_value=0.0, value=50.0, step=5.0)

if st.button("🚀 Solve using Simplex", type="primary"):
    # Build LPP:
    # Min Z = sum(cost_i * x_i)
    # Subject to nutrient >= minimum
    # x_i >= 0
    c = edited["Cost"].to_numpy(dtype=float)
    A = edited[["Protein", "Calories", "Carbs", "Fat"]].to_numpy(dtype=float).T
    b = [protein_req, cal_req, carb_req, fat_req]
    senses = [">=", ">=", ">=", ">="]

    result = solve_lpp(c, A, b, senses)

    if result["status"] == "Optimal":
        st.success("Optimal solution found.")

        sol = pd.DataFrame({
            "Food": edited["Food"],
            "Quantity (servings)": np.round(result["x"], 4)
        })
        sol = sol[sol["Quantity (servings)"] > 1e-6]

        st.subheader("3. Optimal Meal Combination")
        st.dataframe(sol, use_container_width=True)
        st.metric("Minimum Daily Cost (Rs)", f"₹{result['cost']:.2f}")

        totals = edited[["Protein", "Calories", "Carbs", "Fat"]].to_numpy().T @ result["x"]
        nutrition = pd.DataFrame({
            "Nutrient": ["Protein (g)", "Calories", "Carbs (g)", "Fat (g)"],
            "Required": b,
            "Obtained": np.round(totals, 2)
        })
        st.subheader("4. Nutrition Check")
        st.dataframe(nutrition, use_container_width=True)

        st.subheader("5. LPP Formulation")
        var_text = ", ".join([f"x{i+1}" for i in range(len(c))])
        obj = " + ".join([f"{c[i]:g}x{i+1}" for i in range(len(c))])
        st.code(f"Minimize Z = {obj}\n\nSubject to:")
        for r, rhs, nutrient in zip(A, b, ["Protein", "Calories", "Carbs", "Fat"]):
            expr = " + ".join([f"{r[i]:g}x{i+1}" for i in range(len(c))])
            st.write(f"{expr} ≥ {rhs:g}   ({nutrient})")
        st.write("x₁, x₂, …, x₈ ≥ 0")

        st.subheader("6. Simplex Tableaus")
        st.markdown(
            "Phase I finds a feasible basic solution by driving artificial "
            "variables out of the basis. Phase II optimizes the original cost."
        )

        st.markdown("### Phase I")
        display_snapshots(result["phase1"], ["x" + str(i+1) for i in range(8)] + ["e1", "A1", "e2", "A2", "e3", "A3", "e4", "A4"])

        st.markdown("### Phase II")
        display_snapshots(result["phase2"], result["names"])

        st.subheader("7. Conclusion")
        st.write(
            f"The model selects a combination of canteen food servings that satisfies "
            f"the specified minimum nutrition requirements while minimizing cost. "
            f"For the selected targets, the calculated minimum cost is ₹{result['cost']:.2f}."
        )
    else:
        st.error(f"Solver status: {result['status']}")
        if result.get("phase1"):
            st.write("The Phase-I feasibility process is shown below.")
            display_snapshots(result["phase1"], FOODS["Food"].tolist())

st.sidebar.header("Project Mapping")
st.sidebar.write("✓ LPP formulation")
st.sidebar.write("✓ Slack / surplus / artificial variables")
st.sidebar.write("✓ Two-phase Simplex")
st.sidebar.write("✓ NumPy matrix operations")
st.sidebar.write("✓ Pandas tableau display")
st.sidebar.write("✓ Streamlit web application")
st.sidebar.caption("Replace the illustrative food data with measured canteen data before final submission.")
