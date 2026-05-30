import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ==============================================
# Helper Functions
# ==============================================

def normalize_matrix(matrix, benefit_criteria):
    """
    Normalize the decision matrix.
    
    Parameters:
    - matrix: Decision matrix with alternatives as rows and criteria as columns
    - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
    
    Returns:
    - Normalized decision matrix
    """
    normalized = np.zeros_like(matrix, dtype=float)
    
    for j in range(matrix.shape[1]):
        # Vector normalization
        denominator = np.sqrt(np.sum(matrix[:, j]**2))
        
        # Check for zero division
        if denominator == 0:
            normalized[:, j] = 0
        else:
            normalized[:, j] = matrix[:, j] / denominator
    
    return normalized

def calculate_weights_ahp(criteria_comparisons):
    """
    Calculate weights using AHP and check consistency.
    
    Parameters:
    - criteria_comparisons: Pairwise comparison matrix
    
    Returns:
    - weights: Normalized priority vector
    - cr: Consistency ratio (CR < 0.1 is considered acceptable)
    """
    n = criteria_comparisons.shape[0]
    
    # Random consistency index values for different matrix sizes
    ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    # Use eigh for symmetric matrices to ensure real eigenvalues
    try:
        eigenvals, eigenvecs = np.linalg.eigh(criteria_comparisons)
        max_eigenval_index = np.argmax(eigenvals)
        max_eigenval = eigenvals[max_eigenval_index]
        weights = np.real(eigenvecs[:, max_eigenval_index])
    except np.linalg.LinAlgError:
        # Fall back to general eigenvector computation if matrix is not perfectly symmetric
        eigenvals, eigenvecs = np.linalg.eig(criteria_comparisons)
        max_eigenval_index = np.argmax(np.real(eigenvals))
        max_eigenval = np.real(eigenvals[max_eigenval_index])
        weights = np.real(eigenvecs[:, max_eigenval_index])
    
    # Ensure weights are positive
    if np.any(weights < 0):
        weights = -weights
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Calculate Consistency Index and Ratio
    ci = (max_eigenval - n) / (n - 1) if n > 1 else 0
    cr = ci / ri_values[n] if n in ri_values and ri_values[n] > 0 else 0
    
    return weights, cr

# ==============================================
# AHP Implementation
# ==============================================

def ahp(decision_matrix, criteria_comparisons, benefit_criteria):
    """
    Analytical Hierarchy Process for ranking alternatives.
    
    Parameters:
    - decision_matrix: Matrix with alternatives as rows and criteria as columns
    - criteria_comparisons: Pairwise comparison matrix for criteria
    - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
    
    Returns:
    - ranked_alternatives: Indices of alternatives sorted by preference
    - scores: Preference scores for each alternative
    - cr: Consistency ratio for the pairwise comparisons
    """
    # Preprocess data for cost criteria
    processed_matrix = decision_matrix.copy()
    for j in range(decision_matrix.shape[1]):
        if not benefit_criteria[j] and np.any(decision_matrix[:, j] > 0):
            # Invert values for cost criteria (if not zero)
            processed_matrix[:, j] = 1 / decision_matrix[:, j]
            # Handle any infinities from division by zero
            processed_matrix[np.isinf(processed_matrix[:, j]), j] = 0
    
    # Normalize the decision matrix
    normalized_matrix = normalize_matrix(processed_matrix, benefit_criteria)
    
    # Calculate weights and check consistency
    weights, cr = calculate_weights_ahp(criteria_comparisons)
    
    # Calculate weighted scores
    weighted_matrix = normalized_matrix * weights
    
    # Rank alternatives
    scores = weighted_matrix.sum(axis=1)
    ranked_alternatives = np.argsort(-scores)  # Sort in descending order
    
    return ranked_alternatives, scores, cr, weights

# ==============================================
# TOPSIS Implementation
# ==============================================

def topsis(decision_matrix, weights, benefit_criteria):
    """
    Technique for Order of Preference by Similarity to Ideal Solution.
    
    Parameters:
    - decision_matrix: Matrix with alternatives as rows and criteria as columns
    - weights: Weights for each criterion
    - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
    
    Returns:
    - ranked_alternatives: Indices of alternatives sorted by preference
    - relative_closeness: Relative closeness to ideal solution for each alternative
    """
    # Step 1: Normalize the decision matrix using vector normalization
    # This should be independent of benefit/cost criteria
    norm = np.sqrt(np.sum(decision_matrix**2, axis=0))
    normalized_matrix = decision_matrix / norm
    
    # Step 2: Apply weights
    weighted_matrix = normalized_matrix * weights
    
    # Step 3: Determine ideal solutions based on benefit/cost criteria
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])
    
    for j in range(weighted_matrix.shape[1]):
        if benefit_criteria[j]:
            # For benefit criteria, higher is better
            ideal_best[j] = np.max(weighted_matrix[:, j])
            ideal_worst[j] = np.min(weighted_matrix[:, j])
        else:
            # For cost criteria, lower is better
            ideal_best[j] = np.min(weighted_matrix[:, j])
            ideal_worst[j] = np.max(weighted_matrix[:, j])
    
    # Step 4: Calculate separation measures
    S_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    S_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
    
    # Step 5: Calculate relative closeness
    denominator = S_best + S_worst
    relative_closeness = np.zeros_like(S_worst)
    nonzero_indices = denominator > 0
    relative_closeness[nonzero_indices] = S_worst[nonzero_indices] / denominator[nonzero_indices]
    
    # Step 6: Rank alternatives
    ranked_alternatives = np.argsort(-relative_closeness)  # Sort in descending order
    
    return ranked_alternatives, relative_closeness

# ==============================================
# ELECTRE Implementation
# ==============================================

def electre(decision_matrix, weights, benefit_criteria, c_threshold=None, d_threshold=None):
    """
    ELECTRE method for outranking relations.
    
    Parameters:
    - decision_matrix: Matrix with alternatives as rows and criteria as columns
    - weights: Weights for each criterion
    - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
    - c_threshold: Concordance threshold (if None, median value is used)
    - d_threshold: Discordance threshold (if None, median value is used)
    
    Returns:
    - ranked_alternatives: Indices of alternatives sorted by net outranking flow
    - net_flow: Net outranking flow for each alternative
    - concordance_matrix: Concordance indices between alternatives
    - discordance_matrix: Discordance indices between alternatives
    """
    # Normalize the decision matrix
    normalized_matrix = normalize_matrix(decision_matrix, benefit_criteria)
    
    # Apply weights
    weighted_matrix = normalized_matrix * weights
    
    n = weighted_matrix.shape[0]
    concordance_matrix = np.zeros((n, n))
    discordance_matrix = np.zeros((n, n))
    
    # Calculate concordance and discordance matrices
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # Concordance: sum of weights where alternative i is at least as good as j
            concordance_indices = []
            for k in range(len(benefit_criteria)):
                if benefit_criteria[k]:
                    if weighted_matrix[i, k] >= weighted_matrix[j, k]:
                        concordance_indices.append(k)
                else:
                    if weighted_matrix[i, k] <= weighted_matrix[j, k]:
                        concordance_indices.append(k)
            
            concordance_matrix[i, j] = sum(weights[concordance_indices])
            
            # Discordance: maximum normalized difference where j is better than i
            discordance_values = []
            for k in range(len(benefit_criteria)):
                if benefit_criteria[k]:
                    if weighted_matrix[j, k] > weighted_matrix[i, k]:
                        # Calculate normalized difference
                        max_diff = np.max(weighted_matrix[:, k]) - np.min(weighted_matrix[:, k])
                        if max_diff > 0:
                            discordance_values.append((weighted_matrix[j, k] - weighted_matrix[i, k]) / max_diff)
                else:
                    if weighted_matrix[j, k] < weighted_matrix[i, k]:
                        # Calculate normalized difference
                        max_diff = np.max(weighted_matrix[:, k]) - np.min(weighted_matrix[:, k])
                        if max_diff > 0:
                            discordance_values.append((weighted_matrix[i, k] - weighted_matrix[j, k]) / max_diff)
            
            discordance_matrix[i, j] = max(discordance_values) if discordance_values else 0
    
    # Set thresholds if not provided
    if c_threshold is None:
        c_threshold = np.median(concordance_matrix[concordance_matrix > 0])
    if d_threshold is None:
        d_threshold = np.median(discordance_matrix[discordance_matrix > 0])
    
    # Create outranking matrix
    outranking = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and concordance_matrix[i, j] >= c_threshold and discordance_matrix[i, j] <= d_threshold:
                outranking[i, j] = 1  # i outranks j
    
    # Calculate net outranking flow (similar to PROMETHEE)
    outgoing_flow = outranking.sum(axis=1)  # Row sum (outgoing)
    incoming_flow = outranking.sum(axis=0)  # Column sum (incoming)
    net_flow = outgoing_flow - incoming_flow
    
    # Rank alternatives based on net outranking flow
    ranked_alternatives = np.argsort(-net_flow)  # Sort in descending order
    
    return ranked_alternatives, net_flow, concordance_matrix, discordance_matrix

# ==============================================
# Method-specific Fraud Detection Functions
# ==============================================

def detect_fraud_ahp(decision_matrix, criteria_comparisons, benefit_criteria, threshold_factor=0.5):
    """Detect fraud using AHP method with adaptive threshold"""
    ranked_alternatives, scores, cr, weights = ahp(decision_matrix, criteria_comparisons, benefit_criteria)
    
    # Apply adaptive threshold based on mean and standard deviation
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + threshold_factor * std_score
    
    # Identify suspicious transactions
    suspicious_indices = np.where(scores <= threshold)[0]
    
    return {
        'ranked_alternatives': ranked_alternatives,
        'scores': scores,
        'suspicious_indices': suspicious_indices,
        'threshold': threshold,
        'weights': weights,
        'consistency_ratio': cr
    }

def detect_fraud_topsis(decision_matrix, weights, benefit_criteria, suspicious_percentile=25):
    """
    Detect fraud using TOPSIS with percentile-based threshold.
    
    Parameters:
    - suspicious_percentile: e.g., 25 means bottom 25% are suspicious
    """
    ranked_alternatives, scores = topsis(decision_matrix, weights, benefit_criteria)
    
    # Threshold based on bottom percentile of scores
    threshold = np.percentile(scores, suspicious_percentile)
    
    # Lower scores = closer to negative ideal = more suspicious
    suspicious_indices = np.where(scores <= threshold)[0]
    
    return {
        'ranked_alternatives': ranked_alternatives,
        'scores': scores,
        'suspicious_indices': suspicious_indices,
        'threshold': threshold
    }


def detect_fraud_electre(decision_matrix, weights, benefit_criteria, c_threshold=None, d_threshold=None, threshold_factor=0.5):
    """Detect fraud using ELECTRE method with adaptive threshold"""
    ranked_alternatives, net_flow, concordance_matrix, discordance_matrix = electre(
        decision_matrix, weights, benefit_criteria, c_threshold, d_threshold)
    
    # Apply adaptive threshold based on mean and standard deviation
    mean_score = np.mean(net_flow)
    std_score = np.std(net_flow)
    threshold = mean_score - threshold_factor * std_score  # Lower net flow indicates more suspicious
    
    # Identify suspicious transactions
    suspicious_indices = np.where(net_flow <= threshold)[0]
    
    return {
        'ranked_alternatives': ranked_alternatives,
        'scores': net_flow,
        'suspicious_indices': suspicious_indices,
        'threshold': threshold,
        'concordance_matrix': concordance_matrix,
        'discordance_matrix': discordance_matrix
    }

# ==============================================
# Method-specific Fraud Prevention Functions
# ==============================================

def recommend_strategies_ahp(prevention_matrix, criteria_comparisons, benefit_criteria, suspicious_transaction=None, 
                             transaction_data=None, criteria=None, risk_weights=None):
    """Recommend prevention strategies using AHP method"""
    # Rank strategies using AHP
    ranked_strategies, scores, cr, weights = ahp(prevention_matrix, criteria_comparisons, benefit_criteria)
    
    recommendations = []
    for i, idx in enumerate(ranked_strategies):
        recommendations.append({
            'strategy_index': idx,
            'rank': i + 1,
            'score': scores[idx],
            'rationale': ["Based on AHP hierarchy and criteria weights"]
        })
    
    # If a specific transaction is provided, enhance recommendations with transaction-specific insights
    if suspicious_transaction is not None and transaction_data is not None and criteria is not None:
        # Analyze the transaction for risk factors
        risk_factors = analyze_transaction_ahp(transaction_data, criteria, risk_weights, benefit_criteria)
        
        # Refine recommendations based on transaction risk factors
        for rec in recommendations:
            strategy_idx = rec['strategy_index']
            effectiveness = prevention_matrix[strategy_idx]
            
            # Calculate relevance score for this specific transaction
            relevance_score = 0
            for j, factor in enumerate(criteria):
                if factor in risk_factors:
                    factor_risk = risk_factors[factor]['risk_score']
                    strategy_effect = effectiveness[j]
                    relevance_score += factor_risk * strategy_effect
            
            rec['transaction_relevance'] = relevance_score
            rec['transaction_specific_rationale'] = [
                f"Addresses key risk factor: {list(risk_factors.keys())[0]}"
            ]
    
    return {
        'method': 'AHP',
        'recommendations': recommendations,
        'weights': weights,
        'consistency_ratio': cr
    }

def recommend_strategies_topsis(prevention_matrix, weights, benefit_criteria, suspicious_transaction=None, 
                                transaction_data=None, criteria=None, detection_scores=None):
    """Recommend prevention strategies using TOPSIS method"""
    # Rank strategies using TOPSIS
    ranked_strategies, scores = topsis(prevention_matrix, weights, benefit_criteria)
    
    recommendations = []
    for i, idx in enumerate(ranked_strategies):
        recommendations.append({
            'strategy_index': idx,
            'rank': i + 1,
            'score': scores[idx],
            'rationale': ["Based on distance from ideal and negative-ideal solutions"]
        })
    
    # If a specific transaction is provided, enhance recommendations with transaction-specific insights
    if suspicious_transaction is not None and transaction_data is not None and criteria is not None:
        # Analyze the transaction for risk factors
        risk_factors = analyze_transaction_topsis(transaction_data, criteria, weights, benefit_criteria)
        
        # Refine recommendations based on transaction risk factors
        for rec in recommendations:
            strategy_idx = rec['strategy_index']
            effectiveness = prevention_matrix[strategy_idx]
            
            # Calculate relevance score for this specific transaction
            relevance_score = 0
            for j, factor in enumerate(criteria):
                if factor in risk_factors:
                    factor_risk = risk_factors[factor]['distance_factor']
                    strategy_effect = effectiveness[j]
                    relevance_score += factor_risk * strategy_effect
            
            rec['transaction_relevance'] = relevance_score
            rec['transaction_specific_rationale'] = [
                f"Effectively addresses distance from ideal for: {list(risk_factors.keys())[0]}"
            ]
    
    return {
        'method': 'TOPSIS',
        'recommendations': recommendations
    }

def recommend_strategies_electre(prevention_matrix, weights, benefit_criteria, c_threshold=None, d_threshold=None,
                                 suspicious_transaction=None, transaction_data=None, criteria=None, concordance_matrix=None):
    """Recommend prevention strategies using ELECTRE method"""
    # Rank strategies using ELECTRE
    ranked_strategies, net_flow, concordance_matrix, discordance_matrix = electre(
        prevention_matrix, weights, benefit_criteria, c_threshold, d_threshold)
    
    recommendations = []
    for i, idx in enumerate(ranked_strategies):
        recommendations.append({
            'strategy_index': idx,
            'rank': i + 1,
            'net_flow': net_flow[idx],
            'rationale': ["Based on outranking relations and net flow"]
        })
    
    # If a specific transaction is provided, enhance recommendations with transaction-specific insights
    if suspicious_transaction is not None and transaction_data is not None and criteria is not None:
        # Analyze the transaction for risk factors
        risk_factors = analyze_transaction_electre(transaction_data, criteria, weights, benefit_criteria)
        
        # Refine recommendations based on transaction risk factors
        for rec in recommendations:
            strategy_idx = rec['strategy_index']
            effectiveness = prevention_matrix[strategy_idx]
            
            # Calculate relevance score for this specific transaction
            relevance_score = 0
            for j, factor in enumerate(criteria):
                if factor in risk_factors:
                    factor_risk = risk_factors[factor]['discordance_factor']
                    strategy_effect = effectiveness[j]
                    relevance_score += factor_risk * strategy_effect
            
            rec['transaction_relevance'] = relevance_score
            rec['transaction_specific_rationale'] = [
                f"Reduces discordance in: {list(risk_factors.keys())[0]}"
            ]
    
    return {
        'method': 'ELECTRE',
        'recommendations': recommendations,
        'concordance_matrix': concordance_matrix,
        'discordance_matrix': discordance_matrix
    }

# ==============================================
# Transaction Analysis Functions
# ==============================================

def analyze_transaction_ahp(transaction_data, criteria, weights, benefit_criteria):
    """Analyze transaction using AHP methodology to identify risk factors"""
    risk_factors = {}
    
    # Calculate weighted normalized values
    normalized = normalize_matrix(transaction_data.reshape(1, -1), benefit_criteria)
    weighted_values = normalized[0] * weights
    
    # Identify high-risk factors
    for i, criterion in enumerate(criteria):
        if benefit_criteria[i]:
            # For benefit criteria, high values are better (low risk)
            risk_score = 1 - weighted_values[i]
        else:
            # For cost criteria, high values are worse (high risk)
            risk_score = weighted_values[i]
        
        # Calculate contribution to overall risk
        contribution = risk_score * weights[i]
        
        risk_factors[criterion] = {
            'raw_value': transaction_data[i],
            'normalized_weighted_value': weighted_values[i],
            'risk_score': risk_score,
            'contribution': contribution,
            'weight': weights[i],
            'is_benefit': benefit_criteria[i]
        }
    
    # Calculate percentage contribution to risk
    total_contribution = sum([v['contribution'] for v in risk_factors.values()])
    if total_contribution > 0:
        for factor in risk_factors:
            risk_factors[factor]['percentage'] = risk_factors[factor]['contribution'] / total_contribution * 100
    
    # Sort by contribution (highest risk factors first)
    risk_factors = {k: v for k, v in sorted(risk_factors.items(), 
                                        key=lambda item: item[1]['contribution'], 
                                        reverse=True)}
    
    return risk_factors

def analyze_transaction_topsis(transaction_data, criteria, weights, benefit_criteria):
    """Analyze transaction using TOPSIS methodology to identify risk factors"""
    risk_factors = {}
    
    # Calculate normalized values
    normalized = normalize_matrix(transaction_data.reshape(1, -1), benefit_criteria)
    weighted_values = normalized[0] * weights
    
    # Create ideal and anti-ideal points
    ideal_best = np.zeros(len(criteria))
    ideal_worst = np.zeros(len(criteria))
    
    for j in range(len(criteria)):
        if benefit_criteria[j]:
            ideal_best[j] = 1.0  # Max normalized value (theoretical)
            ideal_worst[j] = 0.0  # Min normalized value
        else:
            ideal_best[j] = 0.0  # Min normalized value
            ideal_worst[j] = 1.0  # Max normalized value (theoretical)
    
    # Calculate distances and risk factors
    for i, criterion in enumerate(criteria):
        # Distance to ideal (smaller is better)
        distance_to_ideal = abs(weighted_values[i] - ideal_best[i])
        
        # Distance to anti-ideal (larger is better)
        distance_to_anti_ideal = abs(weighted_values[i] - ideal_worst[i])
        
        # Distance factor (higher means higher risk)
        if benefit_criteria[i]:
            distance_factor = distance_to_ideal / (distance_to_ideal + distance_to_anti_ideal) if (distance_to_ideal + distance_to_anti_ideal) > 0 else 0
        else:
            distance_factor = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal) if (distance_to_ideal + distance_to_anti_ideal) > 0 else 0
        
        risk_factors[criterion] = {
            'raw_value': transaction_data[i],
            'weighted_value': weighted_values[i],
            'distance_to_ideal': distance_to_ideal,
            'distance_to_anti_ideal': distance_to_anti_ideal,
            'distance_factor': distance_factor,
            'weight': weights[i],
            'is_benefit': benefit_criteria[i]
        }
    
    # Sort by distance factor (higher risk first)
    risk_factors = {k: v for k, v in sorted(risk_factors.items(), 
                                        key=lambda item: item[1]['distance_factor'], 
                                        reverse=True)}
    
    return risk_factors

def analyze_transaction_electre(transaction_data, criteria, weights, benefit_criteria):
    """Analyze transaction using ELECTRE methodology to identify risk factors"""
    risk_factors = {}
    
    # Calculate normalized values
    normalized = normalize_matrix(transaction_data.reshape(1, -1), benefit_criteria)
    weighted_values = normalized[0] * weights
    
    # Create reference points for comparison
    # (For a single transaction, we compare to theoretical best/worst)
    reference_points = np.zeros((2, len(criteria)))
    
    for j in range(len(criteria)):
        if benefit_criteria[j]:
            reference_points[0, j] = 1.0  # Best (max value)
            reference_points[1, j] = 0.0  # Worst (min value)
        else:
            reference_points[0, j] = 0.0  # Best (min value)
            reference_points[1, j] = 1.0  # Worst (max value)
    
    # Calculate concordance and discordance for each criterion
    for i, criterion in enumerate(criteria):
        concordance_values = []
        discordance_values = []
        
        # Compare with best reference point
        if benefit_criteria[i]:
            # For benefit criteria
            if weighted_values[i] >= reference_points[0, i]:
                concordance_factor = 1.0
                discordance_factor = 0.0
            else:
                concordance_factor = weighted_values[i] / reference_points[0, i] if reference_points[0, i] > 0 else 0
                discordance_factor = (reference_points[0, i] - weighted_values[i]) / reference_points[0, i] if reference_points[0, i] > 0 else 0
        else:
            # For cost criteria
            if weighted_values[i] <= reference_points[0, i]:
                concordance_factor = 1.0
                discordance_factor = 0.0
            else:
                concordance_factor = reference_points[0, i] / weighted_values[i] if weighted_values[i] > 0 else 0
                discordance_factor = (weighted_values[i] - reference_points[0, i]) / weighted_values[i] if weighted_values[i] > 0 else 0
        
        risk_factors[criterion] = {
            'raw_value': transaction_data[i],
            'weighted_value': weighted_values[i],
            'concordance_factor': concordance_factor,
            'discordance_factor': discordance_factor,
            'risk_score': discordance_factor * weights[i],  # Higher discordance means higher risk
            'weight': weights[i],
            'is_benefit': benefit_criteria[i]
        }
    
    # Sort by risk score (higher risk first)
    risk_factors = {k: v for k, v in sorted(risk_factors.items(), 
                                        key=lambda item: item[1]['risk_score'], 
                                        reverse=True)}
    
    return risk_factors

# ==============================================
# Visualization Functions
# ==============================================

def plot_rankings_comparison(methods, rankings, names, title="Rankings Comparison"):
    """Plot a comparison of rankings from different methods."""
    n_methods = len(methods)
    n_alternatives = len(rankings[0])
    
    plt.figure(figsize=(12, 4))
    
    for i, method in enumerate(methods):
        plt.plot(rankings[i], range(n_alternatives), 'o-', label=method)
    
    plt.yticks(range(n_alternatives), names)
    plt.xlabel("Rank (1 is best)")
    plt.ylabel("Alternatives")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()

def plot_scores(methods, scores, names, title="Scores Comparison"):
    """Plot scores from different methods."""
    n_methods = len(methods)
    n_alternatives = len(scores[0])
    
    plt.figure(figsize=(15, 4))
    
    # Create index for bars
    index = np.arange(n_alternatives)
    width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        plt.bar(index + i * width - 0.4 + width/2, scores[i], width, label=method)
    
    plt.xlabel("Alternatives")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(index, names)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_heatmap(matrix, row_labels, col_labels, title="Heatmap"):
    """Plot a heatmap for a matrix."""
    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, cmap='viridis')
    
    # Add labels
    plt.xticks(range(len(col_labels)), col_labels, rotation=90)
    plt.yticks(range(len(row_labels)), row_labels)
    
    # Add colorbar
    plt.colorbar()
    
    # Add values on the heatmap
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")
    
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def plot_detection_comparison(methods, transaction_names, suspicious_indices_list, scores_list):
    """Plot a comparison of fraud detection results from different methods."""
    n_methods = len(methods)
    n_transactions = len(transaction_names)
    
    # Create a binary matrix indicating suspicious transactions
    detection_matrix = np.zeros((n_methods, n_transactions))
    
    for i, suspicious_indices in enumerate(suspicious_indices_list):
        detection_matrix[i, suspicious_indices] = 1
    
    # Plot heatmap
    plt.figure(figsize=(15, 3))
    plt.imshow(detection_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Add labels
    plt.yticks(range(n_methods), methods)
    plt.xticks(range(n_transactions), transaction_names)
    
    # Add annotations
    for i in range(n_methods):
        for j in range(n_transactions):
            if detection_matrix[i, j] == 1:
                plt.text(j, i, "Suspicious", ha="center", va="center", color="white")
            else:
                plt.text(j, i, "Normal", ha="center", va="center")
    
    plt.title("Fraud Detection Comparison Across Methods")
    plt.xlabel("Transactions")
    plt.ylabel("Methods")
    plt.colorbar(ticks=[0, 1], label="Suspicious Status")
    plt.tight_layout()
    
    return plt.gcf()

def plot_prevention_comparison(methods, strategy_names, recommendations_list):
    """Plot a comparison of prevention strategy rankings from different methods."""
    n_methods = len(methods)
    n_strategies = len(strategy_names)
    
    # Create a matrix of rankings
    rankings_matrix = np.zeros((n_methods, n_strategies))
    
    for i, recommendations in enumerate(recommendations_list):
        for rec in recommendations['recommendations']:
            strategy_idx = rec['strategy_index']
            rank = rec['rank']
            rankings_matrix[i, strategy_idx] = rank
    
    # Plot heatmap
    plt.figure(figsize=(15, 3))
    plt.imshow(rankings_matrix, cmap='YlGnBu', aspect='auto')
    
    # Add labels
    plt.yticks(range(n_methods), methods)

# Continuing the code from where it left off:

    plt.xticks(range(n_strategies), strategy_names)
    
    # Add annotations
    for i in range(n_methods):
        for j in range(n_strategies):
            plt.text(j, i, f"{int(rankings_matrix[i, j])}", ha="center", va="center", color="white")
    
    plt.title("Prevention Strategy Rankings Across Methods")
    plt.xlabel("Strategies")
    plt.ylabel("Methods")
    plt.colorbar(label="Rank (1 is best)")
    plt.tight_layout()
    
    return plt.gcf()

def plot_risk_factors(risk_factors, method_name, title="Risk Factor Analysis"):
    """Plot risk factors identified in a transaction."""
    factors = list(risk_factors.keys())
    
    if method_name == "AHP":
        values = [risk_factors[f]['contribution'] for f in factors]
        metric_name = "Risk Contribution"
    elif method_name == "TOPSIS":
        values = [risk_factors[f]['distance_factor'] for f in factors]
        metric_name = "Distance Factor"
    else:  # ELECTRE
        values = [risk_factors[f]['risk_score'] for f in factors]
        metric_name = "Risk Score"
    
    # Sort factors by values
    sorted_indices = np.argsort(values)
    factors = [factors[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 3))
    plt.barh(factors, values)
    plt.xlabel(metric_name)
    plt.ylabel("Risk Factors")
    plt.title(f"{title} ({method_name})")
    plt.tight_layout()
    
    return plt.gcf()

# ==============================================
# Ensemble Method
# ==============================================

def ensemble_fraud_detection(decision_matrix, criteria_comparisons, weights, benefit_criteria, 
                            transaction_names=None, threshold_factor=0.5):
    """
    Perform fraud detection using an ensemble of AHP, TOPSIS, and ELECTRE methods.
    
    Parameters:
    - decision_matrix: Matrix with transactions as rows and criteria as columns
    - criteria_comparisons: Pairwise comparison matrix for criteria (for AHP)
    - weights: Weights for each criterion (for TOPSIS and ELECTRE)
    - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
    - transaction_names: Names/IDs for each transaction (optional)
    - threshold_factor: Factor to adjust threshold sensitivity
    
    Returns:
    - Dictionary with detection results from each method and ensemble decision
    """
    # Run individual methods
    ahp_results = detect_fraud_ahp(decision_matrix, criteria_comparisons, benefit_criteria, threshold_factor)
    topsis_results = detect_fraud_topsis(decision_matrix, weights, benefit_criteria, threshold_factor)
    electre_results = detect_fraud_electre(decision_matrix, weights, benefit_criteria, 
                                          threshold_factor=threshold_factor)
    
    # Get suspicious indices from each method
    ahp_suspicious = set(ahp_results['suspicious_indices'])
    topsis_suspicious = set(topsis_results['suspicious_indices'])
    electre_suspicious = set(electre_results['suspicious_indices'])
    
    # Create ensemble results with voting
    n_transactions = decision_matrix.shape[0]
    votes = np.zeros(n_transactions)
    
    for idx in ahp_suspicious:
        votes[idx] += 1
    for idx in topsis_suspicious:
        votes[idx] += 1
    for idx in electre_suspicious:
        votes[idx] += 1
    
    # Transactions flagged by majority (at least 2 methods)
    majority_suspicious = np.where(votes >= 2)[0]
    
    # Transactions flagged by all methods
    unanimous_suspicious = np.where(votes == 3)[0]
    
    if transaction_names is None:
        transaction_names = [f"Transaction {i+1}" for i in range(n_transactions)]
    
    return {
        'ahp_results': ahp_results,
        'topsis_results': topsis_results,
        'electre_results': electre_results,
        'majority_suspicious': majority_suspicious,
        'unanimous_suspicious': unanimous_suspicious,
        'transaction_names': transaction_names,
        'votes': votes
    }

def ensemble_prevention_strategies(prevention_matrix, criteria_comparisons, weights, benefit_criteria,
                                  strategy_names=None, suspicious_transaction=None, transaction_data=None, criteria=None):
    """
    Recommend prevention strategies using an ensemble of AHP, TOPSIS, and ELECTRE methods.
    
    Parameters:
    - prevention_matrix: Matrix with strategies as rows and criteria as columns
    - criteria_comparisons: Pairwise comparison matrix for criteria (for AHP)
    - weights: Weights for each criterion (for TOPSIS and ELECTRE)
    - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
    - strategy_names: Names for each prevention strategy (optional)
    - suspicious_transaction: The index of a suspicious transaction to analyze
    - transaction_data: Data for the suspicious transaction
    - criteria: Names of criteria
    
    Returns:
    - Dictionary with prevention strategy recommendations from each method and ensemble decision
    """
    # Run individual methods
    ahp_recommendations = recommend_strategies_ahp(
        prevention_matrix, criteria_comparisons, benefit_criteria,
        suspicious_transaction, transaction_data, criteria, weights)
    
    topsis_recommendations = recommend_strategies_topsis(
        prevention_matrix, weights, benefit_criteria,
        suspicious_transaction, transaction_data, criteria)
    
    electre_recommendations = recommend_strategies_electre(
        prevention_matrix, weights, benefit_criteria,
        suspicious_transaction=suspicious_transaction, 
        transaction_data=transaction_data, 
        criteria=criteria)
    
    # Create ensemble recommendations
    n_strategies = prevention_matrix.shape[0]
    ensemble_scores = np.zeros(n_strategies)
    
    # Assign points based on rank (inverse scoring: rank 1 = n points, rank 2 = n-1 points, etc.)
    for rec in ahp_recommendations['recommendations']:
        ensemble_scores[rec['strategy_index']] += n_strategies - rec['rank'] + 1
    
    for rec in topsis_recommendations['recommendations']:
        ensemble_scores[rec['strategy_index']] += n_strategies - rec['rank'] + 1
    
    for rec in electre_recommendations['recommendations']:
        ensemble_scores[rec['strategy_index']] += n_strategies - rec['rank'] + 1
    
    # Rank strategies based on ensemble scores
    ensemble_ranking = np.argsort(-ensemble_scores)
    
    if strategy_names is None:
        strategy_names = [f"Strategy {i+1}" for i in range(n_strategies)]
    
    # Create ensemble recommendations list
    ensemble_recommendations = []
    for i, idx in enumerate(ensemble_ranking):
        # Find individual method rankings
        ahp_rank = next((rec['rank'] for rec in ahp_recommendations['recommendations'] 
                        if rec['strategy_index'] == idx), None)
        topsis_rank = next((rec['rank'] for rec in topsis_recommendations['recommendations'] 
                          if rec['strategy_index'] == idx), None)
        electre_rank = next((rec['rank'] for rec in electre_recommendations['recommendations'] 
                           if rec['strategy_index'] == idx), None)
        
        ensemble_recommendations.append({
            'strategy_index': idx,
            'strategy_name': strategy_names[idx],
            'rank': i + 1,
            'ensemble_score': ensemble_scores[idx],
            'individual_rankings': {
                'AHP': ahp_rank,
                'TOPSIS': topsis_rank,
                'ELECTRE': electre_rank
            }
        })
    
    return {
        'ahp_recommendations': ahp_recommendations,
        'topsis_recommendations': topsis_recommendations,
        'electre_recommendations': electre_recommendations,
        'ensemble_recommendations': ensemble_recommendations,
        'strategy_names': strategy_names
    }

# ==============================================
# Complete System Integration
# ==============================================

class FraudDetectionSystem:
    """
    A comprehensive system for fraud detection and prevention using multi-criteria decision methods.
    """
    def __init__(self, criteria, criteria_comparisons=None, weights=None, benefit_criteria=None):
        """
        Initialize the fraud detection system.
        
        Parameters:
        - criteria: List of criteria names
        - criteria_comparisons: Pairwise comparison matrix for criteria (for AHP)
        - weights: Weights for each criterion (for TOPSIS and ELECTRE)
        - benefit_criteria: Boolean array indicating if each criterion is a benefit (True) or cost (False)
        """
        self.criteria = criteria
        self.criteria_comparisons = criteria_comparisons
        
        # If weights not provided, calculate from comparisons using AHP
        if weights is None and criteria_comparisons is not None:
            self.weights, self.cr = calculate_weights_ahp(criteria_comparisons)
        else:
            self.weights = weights
            self.cr = None
        
        # Default all criteria to cost if not specified
        if benefit_criteria is None:
            self.benefit_criteria = np.zeros(len(criteria), dtype=bool)
        else:
            self.benefit_criteria = benefit_criteria
        
        # Initialize empty data
        self.transactions = None
        self.transaction_names = None
        self.prevention_strategies = None
        self.strategy_names = None
        
        # Results storage
        self.detection_results = None
        self.prevention_results = None
    
    def load_transaction_data(self, transactions, transaction_names=None):
        """
        Load transaction data for analysis.
        
        Parameters:
        - transactions: Matrix with transactions as rows and criteria as columns
        - transaction_names: Names/IDs for each transaction (optional)
        """
        self.transactions = np.array(transactions)
        
        if transaction_names is None:
            self.transaction_names = [f"Transaction {i+1}" for i in range(self.transactions.shape[0])]
        else:
            self.transaction_names = transaction_names
    
    def load_prevention_strategies(self, prevention_strategies, strategy_names=None):
        """
        Load prevention strategies for recommendation.
        
        Parameters:
        - prevention_strategies: Matrix with strategies as rows and criteria (effectiveness) as columns
        - strategy_names: Names for each prevention strategy (optional)
        """
        self.prevention_strategies = np.array(prevention_strategies)
        
        if strategy_names is None:
            self.strategy_names = [f"Strategy {i+1}" for i in range(self.prevention_strategies.shape[0])]
        else:
            self.strategy_names = strategy_names
    
    def detect_fraud(self, method='ensemble', threshold_factor=0.5):
        """
        Detect fraud using specified method.
        
        Parameters:
        - method: Detection method ('ahp', 'topsis', 'electre', or 'ensemble')
        - threshold_factor: Factor to adjust threshold sensitivity
        
        Returns:
        - Detection results
        """
        if self.transactions is None:
            raise ValueError("Transaction data must be loaded first")
        
        if method.lower() == 'ahp':
            if self.criteria_comparisons is None:
                raise ValueError("Criteria comparisons required for AHP method")
            results = detect_fraud_ahp(self.transactions, self.criteria_comparisons, 
                                      self.benefit_criteria, threshold_factor)
        
        elif method.lower() == 'topsis':
            if self.weights is None:
                raise ValueError("Criteria weights required for TOPSIS method")
            results = detect_fraud_topsis(self.transactions, self.weights, 
                                         self.benefit_criteria, threshold_factor)
        
        elif method.lower() == 'electre':
            if self.weights is None:
                raise ValueError("Criteria weights required for ELECTRE method")
            results = detect_fraud_electre(self.transactions, self.weights, 
                                          self.benefit_criteria, threshold_factor=threshold_factor)
        
        elif method.lower() == 'ensemble':
            if self.criteria_comparisons is None or self.weights is None:
                raise ValueError("Criteria comparisons and weights required for ensemble method")
            results = ensemble_fraud_detection(self.transactions, self.criteria_comparisons, 
                                             self.weights, self.benefit_criteria, 
                                             self.transaction_names, threshold_factor)
        else:
            raise ValueError("Method must be 'ahp', 'topsis', 'electre', or 'ensemble'")
        
        self.detection_results = results
        return results
    
    def recommend_strategies(self, method='ensemble', suspicious_transaction=None):
        """
        Recommend prevention strategies using specified method.
        
        Parameters:
        - method: Recommendation method ('ahp', 'topsis', 'electre', or 'ensemble')
        - suspicious_transaction: Index of suspicious transaction for targeted recommendations
        
        Returns:
        - Prevention strategy recommendations
        """
        if self.prevention_strategies is None:
            raise ValueError("Prevention strategies must be loaded first")
        
        transaction_data = None
        if suspicious_transaction is not None:
            if self.transactions is None:
                raise ValueError("Transaction data must be loaded to provide targeted recommendations")
            transaction_data = self.transactions[suspicious_transaction]
        
        if method.lower() == 'ahp':
            if self.criteria_comparisons is None:
                raise ValueError("Criteria comparisons required for AHP method")
            results = recommend_strategies_ahp(self.prevention_strategies, self.criteria_comparisons, 
                                             self.benefit_criteria, suspicious_transaction, 
                                             transaction_data, self.criteria, self.weights)
        
        elif method.lower() == 'topsis':
            if self.weights is None:
                raise ValueError("Criteria weights required for TOPSIS method")
            results = recommend_strategies_topsis(self.prevention_strategies, self.weights, 
                                                self.benefit_criteria, suspicious_transaction, 
                                                transaction_data, self.criteria)
        
        elif method.lower() == 'electre':
            if self.weights is None:
                raise ValueError("Criteria weights required for ELECTRE method")
            results = recommend_strategies_electre(self.prevention_strategies, self.weights, 
                                                 self.benefit_criteria, suspicious_transaction=suspicious_transaction, 
                                                 transaction_data=transaction_data, criteria=self.criteria)
        
        elif method.lower() == 'ensemble':
            if self.criteria_comparisons is None or self.weights is None:
                raise ValueError("Criteria comparisons and weights required for ensemble method")
            results = ensemble_prevention_strategies(self.prevention_strategies, self.criteria_comparisons, 
                                                  self.weights, self.benefit_criteria, self.strategy_names,
                                                  suspicious_transaction, transaction_data, self.criteria)
        else:
            raise ValueError("Method must be 'ahp', 'topsis', 'electre', or 'ensemble'")
        
        self.prevention_results = results
        return results
    
    def analyze_transaction(self, transaction_index, method='ensemble'):
        """
        Analyze a specific transaction to identify risk factors.
        
        Parameters:
        - transaction_index: Index of the transaction to analyze
        - method: Analysis method ('ahp', 'topsis', 'electre', or 'ensemble')
        
        Returns:
        - Risk factor analysis
        """
        if self.transactions is None:
            raise ValueError("Transaction data must be loaded first")
        
        transaction_data = self.transactions[transaction_index]
        
        if method.lower() == 'ahp':
            if self.weights is None:
                raise ValueError("Criteria weights required for analysis")
            risk_factors = analyze_transaction_ahp(transaction_data, self.criteria, 
                                                 self.weights, self.benefit_criteria)
            method_name = "AHP"
        
        elif method.lower() == 'topsis':
            if self.weights is None:
                raise ValueError("Criteria weights required for analysis")
            risk_factors = analyze_transaction_topsis(transaction_data, self.criteria, 
                                                    self.weights, self.benefit_criteria)
            method_name = "TOPSIS"
        
        elif method.lower() == 'electre':
            if self.weights is None:
                raise ValueError("Criteria weights required for analysis")
            risk_factors = analyze_transaction_electre(transaction_data, self.criteria, 
                                                     self.weights, self.benefit_criteria)
            method_name = "ELECTRE"
        
        elif method.lower() == 'ensemble':
            # Perform analysis with all methods and combine results
            ahp_factors = analyze_transaction_ahp(transaction_data, self.criteria, 
                                               self.weights, self.benefit_criteria)
            topsis_factors = analyze_transaction_topsis(transaction_data, self.criteria, 
                                                     self.weights, self.benefit_criteria)
            electre_factors = analyze_transaction_electre(transaction_data, self.criteria, 
                                                       self.weights, self.benefit_criteria)
            
            return {
                'ahp_analysis': ahp_factors,
                'topsis_analysis': topsis_factors,
                'electre_analysis': electre_factors,
                'transaction_name': self.transaction_names[transaction_index],
                'transaction_data': transaction_data
            }
        else:
            raise ValueError("Method must be 'ahp', 'topsis', 'electre', or 'ensemble'")
        
        return {
            'risk_factors': risk_factors,
            'method': method_name,
            'transaction_name': self.transaction_names[transaction_index],
            'transaction_data': transaction_data
        }
    
    def visualize_detection_results(self):
        """
        Visualize fraud detection results.
        
        Returns:
        - Figure object
        """
        if self.detection_results is None:
            raise ValueError("Run detect_fraud first to generate results")
        
        if 'majority_suspicious' in self.detection_results:
            # Ensemble results visualization
            methods = ['AHP', 'TOPSIS', 'ELECTRE']
            suspicious_indices_list = [
                self.detection_results['ahp_results']['suspicious_indices'],
                self.detection_results['topsis_results']['suspicious_indices'],
                self.detection_results['electre_results']['suspicious_indices']
            ]
            scores_list = [
                self.detection_results['ahp_results']['scores'],
                self.detection_results['topsis_results']['scores'],
                self.detection_results['electre_results']['scores']
            ]
            
            return plot_detection_comparison(methods, self.transaction_names, 
                                           suspicious_indices_list, scores_list)
        else:
            # Single method visualization
            if 'scores' in self.detection_results:
                # Create scatter plot of scores with threshold
                plt.figure(figsize=(12, 6))
                plt.scatter(range(len(self.detection_results['scores'])), 
                           self.detection_results['scores'], 
                           c=['red' if i in self.detection_results['suspicious_indices'] else 'blue' 
                             for i in range(len(self.detection_results['scores']))])
                
                plt.axhline(y=self.detection_results['threshold'], color='r', linestyle='--', 
                           label=f"Threshold: {self.detection_results['threshold']:.4f}")
                
                plt.xlabel("Transaction Index")
                plt.ylabel("Score")
                plt.title("Fraud Detection Results")
                plt.legend()
                plt.grid(True)
                plt.xticks(range(len(self.detection_results['scores'])), self.transaction_names, rotation=45)
                plt.tight_layout()
                
                return plt.gcf()
    
    def visualize_prevention_results(self):
        """
        Visualize prevention strategy recommendations.
        
        Returns:
        - Figure object
        """
        if self.prevention_results is None:
            raise ValueError("Run recommend_strategies first to generate results")
        
        if 'ensemble_recommendations' in self.prevention_results:
            # Ensemble results visualization
            methods = ['AHP', 'TOPSIS', 'ELECTRE', 'Ensemble']
            recommendations_list = [
                self.prevention_results['ahp_recommendations'],
                self.prevention_results['topsis_recommendations'],
                self.prevention_results['electre_recommendations'],
                {'recommendations': self.prevention_results['ensemble_recommendations']}
            ]
            
            return plot_prevention_comparison(methods, self.strategy_names, recommendations_list)
        else:
            # Single method visualization
            method = self.prevention_results['method']
            recommendations = self.prevention_results['recommendations']
            
            # Sort recommendations by rank
            sorted_indices = [rec['strategy_index'] for rec in 
                             sorted(recommendations, key=lambda x: x['rank'])]
            
            if method == 'AHP':
                scores = [next((rec['score'] for rec in recommendations 
                              if rec['strategy_index'] == idx), 0) for idx in sorted_indices]
                score_label = "AHP Score"
            elif method == 'TOPSIS':
                scores = [next((rec['score'] for rec in recommendations 
                              if rec['strategy_index'] == idx), 0) for idx in sorted_indices]
                score_label = "Closeness to Ideal"
            else:  # ELECTRE
                scores = [next((rec['net_flow'] for rec in recommendations 
                              if rec['strategy_index'] == idx), 0) for idx in sorted_indices]
                score_label = "Net Flow"
            
            strategy_labels = [self.strategy_names[idx] for idx in sorted_indices]
            
            plt.figure(figsize=(10, 6))
            plt.bar(strategy_labels, scores)
            plt.xlabel("Prevention Strategy")
            plt.ylabel(score_label)
            plt.title(f"Prevention Strategies Ranked by {method}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return plt.gcf()
    
    def visualize_risk_analysis(self, analysis_results):
        """
        Visualize risk factor analysis for a transaction.
        
        Parameters:
        - analysis_results: Results from analyze_transaction method
        
        Returns:
        - Figure object
        """
        if 'ahp_analysis' in analysis_results:
            # Ensemble analysis visualization - show multiple plots
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            
            # AHP analysis
            risk_factors = analysis_results['ahp_analysis']
            factors = list(risk_factors.keys())
            values = [risk_factors[f]['contribution'] for f in factors]
            axes[0].barh(factors, values)
            axes[0].set_xlabel("Risk Contribution")
            axes[0].set_ylabel("Risk Factors")
            axes[0].set_title("AHP Risk Analysis")
            
            # TOPSIS analysis
            risk_factors = analysis_results['topsis_analysis']
            factors = list(risk_factors.keys())
            values = [risk_factors[f]['distance_factor'] for f in factors]
            axes[1].barh(factors, values)
            axes[1].set_xlabel("Distance Factor")
            axes[1].set_ylabel("Risk Factors")
            axes[1].set_title("TOPSIS Risk Analysis")
            
            # ELECTRE analysis
            risk_factors = analysis_results['electre_analysis']
            factors = list(risk_factors.keys())
            values = [risk_factors[f]['risk_score'] for f in factors]
            axes[2].barh(factors, values)
            axes[2].set_xlabel("Risk Score")
            axes[2].set_ylabel("Risk Factors")
            axes[2].set_title("ELECTRE Risk Analysis")
            
            plt.tight_layout()
            return fig
        else:
            # Single method visualization
            risk_factors = analysis_results['risk_factors']
            method = analysis_results['method']
            transaction_name = analysis_results['transaction_name']
            
            return plot_risk_factors(risk_factors, method, 
                                    f"Risk Analysis for {transaction_name}")
    
    def compare_methods(self):
        """
        Compare results from different methods.
        
        Returns:
        - Dictionary with comparison results
        """
        # Run all methods for detection
        ahp_detection = self.detect_fraud(method='ahp')
        topsis_detection = self.detect_fraud(method='topsis')
        electre_detection = self.detect_fraud(method='electre')
        
        # Run all methods for prevention
        suspicious_idx = None
        if len(ahp_detection['suspicious_indices']) > 0:
            suspicious_idx = ahp_detection['suspicious_indices'][0]
        
        ahp_prevention = self.recommend_strategies(method='ahp', suspicious_transaction=suspicious_idx)
        topsis_prevention = self.recommend_strategies(method='topsis', suspicious_transaction=suspicious_idx)
        electre_prevention = self.recommend_strategies(method='electre', suspicious_transaction=suspicious_idx)
        
        # Create method comparison data
        detection_comparison = {
            'methods': ['AHP', 'TOPSIS', 'ELECTRE'],
            'suspicious_counts': [
                len(ahp_detection['suspicious_indices']),
                len(topsis_detection['suspicious_indices']),
                len(electre_detection['suspicious_indices'])
            ],
            'suspicious_indices': [
                ahp_detection['suspicious_indices'],
                topsis_detection['suspicious_indices'],
                electre_detection['suspicious_indices']
            ],
            'agreement_matrix': np.zeros((3, 3))
        }
        
        # Calculate agreement between methods
        methods = [
            set(ahp_detection['suspicious_indices']),
            set(topsis_detection['suspicious_indices']),
            set(electre_detection['suspicious_indices'])
        ]
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    detection_comparison['agreement_matrix'][i, j] = 1.0
                else:
                    # Jaccard index for set similarity
                    intersection = len(methods[i].intersection(methods[j]))
                    union = len(methods[i].union(methods[j]))
                    detection_comparison['agreement_matrix'][i, j] = intersection / union if union > 0 else 1.0
        
        # Compare prevention strategy rankings
        prevention_comparison = {
            'methods': ['AHP', 'TOPSIS', 'ELECTRE'],
            'top_strategy': [
                ahp_prevention['recommendations'][0]['strategy_index'],
                topsis_prevention['recommendations'][0]['strategy_index'],
                electre_prevention['recommendations'][0]['strategy_index']
            ],
            'rank_correlation': np.zeros((3, 3))
        }
        
        # Calculate rank correlation between methods
        ahp_ranks = {rec['strategy_index']: rec['rank'] for rec in ahp_prevention['recommendations']}
        topsis_ranks = {rec['strategy_index']: rec['rank'] for rec in topsis_prevention['recommendations']}
        electre_ranks = {rec['strategy_index']: rec['rank'] for rec in electre_prevention['recommendations']}
        
        ranks = [ahp_ranks, topsis_ranks, electre_ranks]
        n = len(self.strategy_names)
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    prevention_comparison['rank_correlation'][i, j] = 1.0
                else:
                    # Spearman's rank correlation
                    di_squared = 0
                    for idx in range(n):
                        if idx in ranks[i] and idx in ranks[j]:
                            di_squared += (ranks[i][idx] - ranks[j][idx]) ** 2
                    
                    # Calculate correlation coefficient
                    rho = 1 - (6 * di_squared) / (n * (n**2 - 1))
                    prevention_comparison['rank_correlation'][i, j] = rho
        
        return {
            'detection_comparison': detection_comparison,
            'prevention_comparison': prevention_comparison
        }

    def visualize_method_comparison(self, comparison_results):
        """
        Visualize comparison of different methods.
        
        Parameters:
        - comparison_results: Results from compare_methods method
        
        Returns:
        - Dictionary with figure objects
        """
        figures = {}
        
        # Plot detection agreement heatmap
        detection = comparison_results['detection_comparison']
        fig1 = plot_heatmap(detection['agreement_matrix'], 
                          detection['methods'], 
                          detection['methods'],
                          "Agreement in Fraud Detection (Jaccard Index)")
        figures['detection_agreement'] = fig1
        
        # Plot prevention rank correlation heatmap
        prevention = comparison_results['prevention_comparison']
        fig2 = plot_heatmap(prevention['rank_correlation'],
                          prevention['methods'],
                          prevention['methods'],
                          "Correlation in Prevention Strategy Rankings")
        figures['prevention_correlation'] = fig2
        
        # Plot suspicious counts comparison
        fig3, ax = plt.subplots(figsize=(8, 3))
        ax.bar(detection['methods'], detection['suspicious_counts'])
        ax.set_xlabel("Method")
        ax.set_ylabel("Number of Suspicious Transactions")
        ax.set_title("Suspicious Transaction Count by Method")
        figures['suspicious_counts'] = fig3
        
        return figures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(ground_truth, predictions, method_names):
    data = {"Method": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}
    for method_name, pred_indices in zip(method_names, predictions):
        pred = [1 if i in pred_indices else 0 for i in range(len(ground_truth))]
        acc = accuracy_score(ground_truth, pred)
        prec = precision_score(ground_truth, pred, zero_division=0)
        rec = recall_score(ground_truth, pred, zero_division=0)
        f1 = f1_score(ground_truth, pred, zero_division=0)

        data["Method"].append(method_name)
        data["Accuracy"].append(round(acc, 4))
        data["Precision"].append(round(prec, 4))
        data["Recall"].append(round(rec, 4))
        data["F1 Score"].append(round(f1, 4))
    return data

