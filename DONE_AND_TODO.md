# Done

- scripts `teaching/imitation/teaching_bc.py` and `teaching/imitation/teaching_dagger.py` 
are running but don't give satisfactory results 
(reward from the novice way lower than reward from the expert)

- `test_predict_one_step` not passing since new predict vectorized method [05-09-22]

- Check the efficiency of the Conservative Sampling teacher in normal context 
and in BC context [05-09-22]

- Tried dagger on teaching [05-09-22]

# Todo

- Solve bug of "double" environments: expert not working as expected 
if created with in a different environment than the one used to do steps in [?]

- Test not passing (@Julien) [21-10-22]


- Check with other environments that nothing is breaking when using dagger [?]


- Check the computation of the probability of recall (which timestep is considered) 
  => separate update of internal parameters (eg `delta_time`) and query of the recall probabilities

- SILENT BUG: __setattr__ state

