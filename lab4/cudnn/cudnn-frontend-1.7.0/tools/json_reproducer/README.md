## Json reproducer

### Usage

```
usage: json_parser.py [-h] -i INPUT_FILE [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file name
  -v, --verbose         Set logging level to max
```


### Notes

Input is a json representation of graph before validate is called. 

For c++ users, this json can be generated by calling `std::cout << graph << std::endl;` or `graph.print()`.

For python users, this is called by calling the `print(graph)` method.