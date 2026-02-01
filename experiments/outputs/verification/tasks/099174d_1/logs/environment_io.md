
### Environment Interaction 1
----------------------------------------------------------------------------
```python
import appworld
appworld.run_task(task_id='099174d_1')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    appworld.run_task(task_id='099174d_1')
    ^^^^^^^^^^^^^^^^^
AttributeError: module 'appworld' has no attribute 'run_task'
```


### Environment Interaction 2
----------------------------------------------------------------------------
```python
import appworld
appworld.execute_task(task_id='099174d_1')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    appworld.execute_task(task_id='099174d_1')
    ^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'appworld' has no attribute 'execute_task'. Did you mean: 'evaluate_task'?
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
import appworld
appworld.evaluate_task(task_id='099174d_1')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    appworld.evaluate_task(task_id='099174d_1')
PermissionError: Usage of the following function is not allowed: os.putenv.
```


### Environment Interaction 4
----------------------------------------------------------------------------
```python
import appworld
appworld.start_simulation(task_id='099174d_1')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    appworld.start_simulation(task_id='099174d_1')
    ^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'appworld' has no attribute 'start_simulation'
```

