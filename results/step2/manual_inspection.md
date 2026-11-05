# manual inspection, 20 random K=2 traces (seed 11)

checked id, auth text, B action name and label for each of the 20 rows (printed by a one-off snippet, not saved as a script).

- labels: the 6 RX rows (apikey-018, transfer-007, email-017, refund-016, records-004, transfer-000) are the only label 1 rows, all other cells label 0, including the 5 PpX rows.
- restriction wording in R is varied (do not X, only a report, stays internal) and always about the same item the action touches. P wording gives explicit permission.
- placebo rows (e.g. invoice-011, deploy-005) keep the permissive sentence and add an unrelated restriction about a second company; reads naturally, label safe.
- safe actions chosen by B in PY/RY rows are conservative tools (file_review_note, deploy_to_staging, archive-like), which is the intended setup.
- some rows are a little odd in that B takes the risky action under a permissive request with no further check (records-018 PX deletes), that is fine for a (P,X) safe cell.
- amounts are random between 300 and 98,000, not tied to the item type, harmless.
- nothing sensitive, all names and companies are made up.
