# DSA 4060 – Week 3: Content-Based Recommender

**Student ID:** 670444
**Course:** DSA 4060 – Recommender Systems
**Lab:** Week 3 Practical – Building a Simple Content-Based Recommender

A content-based movie recommender built from **structured binary genre features** and a
single user's ratings. TF-IDF and cosine similarity are deliberately **not** used — they
are Week 4 topics. Scoring here is a rating-weighted dot product over five genre flags.

---

## Repository contents

| File | Description |
|---|---|
| `content_based_lab.ipynb` | The complete lab: all 28 steps, with code, outputs and written answers |
| `movies.csv` | The movie dataset — 11 movies × 5 binary genre features |
| `README.md` | This file |

The notebook is committed **with outputs already executed**, so it can be read end to end
without running anything.

---

## How to run

Requires Python 3 and pandas (no other dependencies).

```bash
pip install pandas jupyter
jupyter notebook content_based_lab.ipynb
```

Then **Kernel → Restart & Run All**.

The notebook is self-contained and idempotent: it builds the 10-movie dataset inline
(Step 3), writes `movies.csv`, reloads it, and later appends the unique 11th movie and
re-saves. A clean run therefore always ends with the same 11-row `movies.csv`, whatever
the file contained beforehand.

---

## The dataset

11 movies, each described by five binary features where `1` = genre present, `0` = absent.

| movie_id | title | Action | Comedy | Drama | Romance | SciFi |
|---|---|---|---|---|---|---|
| 1 | Galaxy Mission | 1 | 0 | 0 | 0 | 1 |
| 2 | Love in Nairobi | 0 | 0 | 1 | 1 | 0 |
| 3 | Final Battle | 1 | 0 | 0 | 0 | 0 |
| 4 | Family Weekend | 0 | 1 | 0 | 0 | 0 |
| 5 | Future World | 1 | 0 | 0 | 0 | 1 |
| 6 | Broken Hearts | 0 | 0 | 1 | 1 | 0 |
| 7 | City Detectives | 1 | 0 | 1 | 0 | 0 |
| 8 | Space Warriors | 1 | 0 | 0 | 0 | 1 |
| 9 | Wedding Weekend | 0 | 1 | 0 | 1 | 0 |
| 10 | Hidden Truth | 0 | 0 | 1 | 0 | 0 |
| **11** | **Money Heist** | **1** | **0** | **1** | **1** | **0** |

Movie 11 is my **unique contribution** (Step 24): **Money Heist** (*La Casa de Papel*, 2017).
It is coded Action (the heist and police standoffs), Drama (the crew backstories and hostage
negotiations) and Romance (the Tokyo/Rio and Professor/Raquel subplots). Its feature
combination `[1, 0, 1, 1, 0]` does not appear anywhere else in the dataset; the notebook
verifies this programmatically rather than by eye.

Note that Money Heist is a multi-season **television series**, not a feature film. The
dataset has no column for item format, so the model scores it exactly as it would a
90-minute movie — a gap discussed in the limitations below.

---

## Method

```
Item Metadata → Item Profiles → User Ratings → User Profile
  → Candidate Scores → Remove Seen Items → Rank Items → Top-N Recommendations
```

1. **Item-feature matrix** — slice the five genre columns away from the identifiers.
2. **User profile** — multiply each rated movie's feature row by its rating
   (`rated_features.mul(user_ratings, axis=0)`), then sum down the columns, so highly
   rated movies contribute more.
3. **Normalize** — divide by the total, turning the profile into proportions summing to 1.
4. **Score** — weighted dot product: `movie_features.mul(user_profile).sum(axis=1)`.
5. **Filter** — drop movies whose `movie_id` already appears in the user's ratings.
6. **Rank** — sort descending and take the top *n*.

Steps 2–6 are packaged as `recommend_movies(movies, ratings, features, n=3)`, which copies
the movie table so scoring one user never mutates shared state. A helper,
`user_profile_for(...)`, exposes the profile itself for the interpretation questions.

---

## Results

### User A — ratings: Galaxy Mission 5, Love in Nairobi 2, Final Battle 4, Future World 5

Profile: **Action 0.500, SciFi 0.357, Drama 0.071, Romance 0.071, Comedy 0.000**

| Rank | Movie | Score | Why |
|---|---|---|---|
| 1 | Space Warriors | 0.857 | Action + SciFi — matches both top preferences |
| 2 | City Detectives | 0.571 | Action (0.500) + Drama (0.071) |
| 3 | Broken Hearts | 0.143 | Only matches the two weakest preferences |

### User B — ratings: Love in Nairobi 5, Family Weekend 4, Broken Hearts 5, Wedding Weekend 4

Profile: **Romance 0.438, Drama 0.313, Comedy 0.250, Action 0.000, SciFi 0.000**

Top recommendations: **City Detectives (0.3125)** and **Hidden Truth (0.3125)** — an exact
tie. User A and User B share **no** recommendations: because User B's Action and SciFi
weights are exactly 0, every pure Action/SciFi film scores 0.000 for them, including
User A's best pick.

### My own user — ratings: Galaxy Mission 5, City Detectives 4, Hidden Truth 4, Money Heist 5

Profile: **Action 0.378, Drama 0.351, Romance 0.135, SciFi 0.135**, Comedy 0.000

| Rank | Movie | Score | Matching features |
|---|---|---|---|
| 1 | Future World | 0.514 | Action + SciFi |
| 2 | Space Warriors | 0.514 | Action + SciFi |
| 3 | Love in Nairobi | 0.486 | Drama + Romance |
| 4 | Broken Hearts | 0.486 | Drama + Romance |
| 5 | Final Battle | 0.378 | Action |

Four of these five sit in **tied pairs** (ranks 1–2 and 3–4), and the gap between the two
pairs is just 0.027 — see the limitations below.

---

## Key limitations identified

1. **Only five binary features, so scores collide constantly.** Ties are the norm, not an
   edge case: User B's top two tie exactly, and my own user's Top 5 contains two tied
   pairs. When scores tie, pandas' sort order decides the ranking, not evidence.
2. **Low ratings cannot express dislike.** The profile is a sum of positive
   *rating × feature* terms, so a 1-star review still pushes a genre's weight **up**.
   User A rated *Love in Nairobi* 2/5 and was still recommended *Broken Hearts*.
3. **Cold start.** A user with no ratings divides by zero and gets a `NaN` profile. One
   rating is not enough either — with a single rated movie, normalization cancels the
   rating value entirely, so 5 stars and 1 star yield an identical profile.
4. **Over-specialisation.** A movie can only score above zero on genres the user has
   already rated, so the model cannot produce serendipity and narrows exposure over time.
5. **Rich metadata ignored.** No actors, directors, year, language or plot. *Future World*
   and *Space Warriors* have identical feature vectors and are literally indistinguishable
   to the model — which is exactly why they tie for first place above. There is not even a
   column for item **format**, so Money Heist (a 15-hour series) is scored identically to a
   90-minute film.
6. **Garbage in, garbage out.** Poor or inconsistent tagging produces poor recommendations,
   with no popularity or quality signal to fall back on.

---

## Final question

> *If two users have watched exactly the same movies but gave them different ratings,
> should this recommender produce the same recommendations for both users?*

**No — and it does not.** The notebook demonstrates this with two users given the *same*
four movies and different ratings. Because ratings are the weights in the profile sum, the
profiles diverge and the rankings reorder, **including at the top**: User X's best pick is
*Space Warriors* (0.857) while User Y's is *Money Heist* (0.800), and *Space Warriors*
itself falls from 1st for User X to 4th for User Y (0.500).

The demonstration also exposes two flaws worth noting. *Money Heist* ranks 1st or 2nd for
**both** users, because carrying three of the five features lets it collect weight from
almost any profile — a structural advantage of richly tagged items, not evidence of a
better match. And User Y, who rated both Action films 2/5, still ends up with **Action as
their highest weight (0.300)**, because the model reads a low rating as mild approval. It
reacts to rating differences in the right direction but at nothing like the right
magnitude; mean-centring the ratings would fix this.

---

## Mapping to the lab steps

| Lab steps | Notebook section |
|---|---|
| 1–2 | 1. Import libraries |
| 3 | 2. Create the movie dataset |
| 4–5 | 3. Understand and explore the dataset |
| 6 | 4. Identify the structured features |
| 7 | 5. Inspect one item profile |
| 8–10 | 6. Create user ratings (User A) |
| 11–15 | 7. Build the user profile |
| 16–17 | 8. Calculate recommendation scores |
| 18 | 9. Remove movies already watched |
| 19–20 | 10. Generate the Top 3 recommendations |
| 21 | 11. Build a reusable function |
| 22 | 12. Test a second user (User B) |
| 23 | 13. Thinking exercise — User C |
| 24 | 14. Add my unique movie |
| 25 | 15. Create my own user |
| 26 | 16. Interpretation |
| 27 | 17. Limitations |
| 28 + Final Question | 18. Conclusion |
