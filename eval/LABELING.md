# Labeling guide

Hand labels are the ground truth for `eval/compare.py`. Label from the headlines and price move that
`python -m eval.label` shows - never look at Jev's or Gemini's answers first, or the labels drift toward
the models and accuracy stops meaning anything.

## Sentiment

The overall tone of the company's recent news coverage:

| Key | Label | When |
|---|---|---|
| `u` | bullish | The headlines skew positive for the company |
| `d` | bearish | The headlines skew negative for the company |
| `n` | neutral | The headlines are mixed, non-committal, or there is no recent coverage |

## Material event

**Yes** only if at least one headline reports one of these, happening at this company:

1. An earnings surprise (results clearly above or below expectations)
2. A guidance change (raised, cut, or withdrawn)
3. A merger or acquisition (announced, completed, or called off)
4. Major litigation (a significant lawsuit, verdict, or settlement)
5. Regulatory action aimed at the company (e.g. an FDA decision, an antitrust case, a fine)
6. An executive change (CEO, CFO, or chair)

This is the same list Jev's `material_event` question uses, so the labels measure exactly what Jev
is asked.

**Not material** - common cases that look like news but aren't on the list:

- An earnings *date* or conference call being scheduled
- A routine dividend declaration
- An analyst rating, price target, or new coverage
- The stock hitting a high or low, or price-move recaps
- Comparisons and opinion ("X vs. Y: which to buy?", "3 stocks to watch")
- Market-size reports or articles that only mention the company in passing
- Industry-wide news that doesn't single the company out
- Things outside the six categories, even if sizeable: large orders or contracts, layoffs or
  restructuring, buybacks or dividend changes, product launches

**Borderline?** Ask "is this clearly one of the six?" If you have to argue for it, answer **no**.
Judge the same kind of news the same way every time.

Events anywhere in the headline window count, even if a few days old - that is the window Jev sees.
