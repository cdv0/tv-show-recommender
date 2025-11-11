import pandas as pd
import numpy as np

CSV_FILENAME = "sample-ratings.csv"
TARGET_USER = "Cathleen"


def load_ratings(csv_filename: str) -> pd.DataFrame:
    """
    Load the long-format ratings CSV with columns: user, item, rating.
    Returns a user–item matrix (rows = users, columns = items).
    Ex. Cathleen Jane the Virgin 9
        Alex     Stranger Things 8
    """
    ratings = pd.read_csv(csv_filename)

    expected_cols = {"user", "item", "rating"}
    if not expected_cols.issubset(ratings.columns):
        raise ValueError(
            f"CSV must contain columns {expected_cols}, found {set(ratings.columns)}"
        )

    # Convert ratings table into a matrix
    user_item = ratings.pivot_table(
        index="user",
        columns="item",
        values="rating",
        aggfunc="mean"  # If there are duplicate user-item pairs, average them
    )

    return user_item


def compute_item_similarity(user_item: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item–item cosine similarity based on the user–item rating matrix.
    Returns an item–item similarity DataFrame.
    Ex. user     Jane the Virgin  Stranger Things ...
        Cathleen        9                8
        Alex           NaN/0             9
    """
    # Fill NaNs with 0
    R_filled = user_item.fillna(0.0)

    # Items x users matrix (each row is an item vector)
    M = R_filled.to_numpy().T  # shape: (num_items, num_users)
    item_names = R_filled.columns

    # Compute L2 norms for each item vector
    norms = np.linalg.norm(M, axis=1, keepdims=True)

    # Avoid division by zero by using where=norms!=0
    M_normalized = np.divide(
        M,
        norms,
        out=np.zeros_like(M),
        where=norms != 0
    )

    # Cosine similarity: normalized dot product
    sim_matrix = M_normalized @ M_normalized.T

    item_sim = pd.DataFrame(sim_matrix, index=item_names, columns=item_names)
    return item_sim


def predict_for_user(
    user_id: str,
    user_item: pd.DataFrame,
    item_sim: pd.DataFrame,
    min_sim_threshold: float = 0.0
) -> pd.Series:
    """
    Predict ratings for all items that `user_id` has not rated yet
    using item–item collaborative filtering.

    Returns a Series: index = item, value = predicted_rating,
    sorted from highest to lowest.
    """
    if user_id not in user_item.index:
        raise ValueError(f"User '{user_id}' not found in ratings matrix.")

    user_ratings = user_item.loc[user_id]  # row: one rating per item (or NaN)

    predictions = {}

    # Loop over all items
    for item in user_item.columns:
        # Skip items the user has already rated
        if not np.isnan(user_ratings[item]):
            continue

        # Similarities between this item and all other items
        sims = item_sim.loc[item]

        # Only consider items the user *has* rated
        rated_mask = ~user_ratings.isna()
        sims = sims[rated_mask]
        rated_vals = user_ratings[rated_mask]

        # Optionally ignore very low or negative similarities
        if min_sim_threshold is not None and min_sim_threshold > 0:
            high_sim_mask = sims >= min_sim_threshold
            sims = sims[high_sim_mask]
            rated_vals = rated_vals[high_sim_mask]

        if len(sims) == 0:
            continue

        sim_sum = sims.abs().sum()
        if sim_sum == 0:
            continue

        # Weighted average of the user's ratings for similar items
        weighted_sum = (sims * rated_vals).sum()
        predicted_rating = weighted_sum / sim_sum

        predictions[item] = predicted_rating

    if not predictions:
        return pd.Series(dtype=float, name="predicted_rating")

    pred_series = pd.Series(predictions, name="predicted_rating")
    return pred_series.sort_values(ascending=False)


def main():
    print(f"Loading ratings from '{CSV_FILENAME}'...")
    user_item = load_ratings(CSV_FILENAME)
    print("\nUser–Item Rating Matrix (NaN = not rated):")
    print(user_item)

    print("\nComputing item–item cosine similarity...")
    item_sim = compute_item_similarity(user_item)

    print("\nItem–Item Similarity Matrix (rounded to 2 decimals):")
    print(item_sim.round(2))

    print(f"\nPredicting ratings for user '{TARGET_USER}'...")
    preds = predict_for_user(TARGET_USER, user_item, item_sim, min_sim_threshold=0.0)

    if preds.empty:
        print(
            f"\nNo items to predict for '{TARGET_USER}'. "
            f"They may have rated every item in the dataset."
        )
    else:
        rec_df = preds.reset_index().rename(columns={"index": "item"})
        print(f"\nRanked Recommendations for '{TARGET_USER}':")
        print(rec_df)


if __name__ == "__main__":
    main()
