
### Environment Interaction 1
----------------------------------------------------------------------------
```python
from appworld import AppWorld
aw = AppWorld()
aw.run_task(task_id='13547f5_3')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    aw = AppWorld()
         ^^^^^^^^^^
TypeError: AppWorld.__init__() missing 1 required positional argument: 'task_id'
```


### Environment Interaction 2
----------------------------------------------------------------------------
```python
from appworld import AppWorld
aw = AppWorld(task_id='13547f5_3')
aw.run_task(task_id='13547f5_3')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    aw = AppWorld(task_id='13547f5_3')
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: Usage of the following function is not allowed: os.putenv.
```


### Environment Interaction 3
----------------------------------------------------------------------------
```python
from appworld import AppWorld
aw = AppWorld(task_id='13547f5_3')
aw.run_task(task_id='13547f5_3')
```

```
Execution failed. Traceback:
  File "<python-input>", line 2, in <module>
    aw = AppWorld(task_id='13547f5_3')
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: Usage of the following function is not allowed: os.putenv.
```

