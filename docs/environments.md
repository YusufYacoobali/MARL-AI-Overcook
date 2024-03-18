## Environments and Recipes

Please refer to our [Overcooked Design Document](design.md) for more information on environment/task customization and observation/action spaces.

Our environments and recipes are combined into one argument. Here are the environment
The recipes and map size are defined in the input the user enters however there are four types of maps the user can choose from.

# Mandatory Collaboration

This map type produces layouts where one chef cannot do everything by themselves, they have to work together with another agent to complete the level

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --grid-type t ...other args...`

<p align="center">
<img src="/images/mandatory.png" width=300></img>
</p>

# Optional Collaboration

This type produces a layout where it is possible for one chef to do everything by themselves.

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --grid-type o ...other args...`

<p align="center">
<img src="/images/optional.png" width=300></img>
</p>

# Spread Layout

This type places special counters close to each other

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --grid-type s ...other args...`

<p align="center">
<img src="/images/spread.png" width=300></img>
</p>

# Randomised

A completely random map is produced

To run this environment on our available recipes (tomato, tomato-lettuce, and salad), run `main.py` with the following argument:

`python3 main.py --grid-type r ...other args...`

<p align="center">
<img src="/images/random.png" width=300></img>
</p>
