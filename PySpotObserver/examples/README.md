# Examples

These examples assume you are running from the `PySpotObserver` project root:

```bash
cd PySpotObserver
```

Install the package and dependencies first:

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

The examples load `examples/config_example.yaml` by default. Fill in at least:

```yaml
robot_ip: "192.168.80.3"
username: ""
password: ""
```

You can also override config values on the command line:

```bash
python examples/basic_streaming.py --robot-ip 192.168.80.3 --username <user> --password <password>
```

## Streaming Example

`basic_streaming.py` is the main example. It supports:

- synchronous streaming
- asynchronous streaming with `--async-mode`
- one or two stream configurations, mirrored across one or two robots
- optional OpenCV display
- optional timing summaries with `--print-timing`

Show the full CLI:

```bash
python examples/basic_streaming.py --help
```

Run a basic single-stream session:

```bash
python examples/basic_streaming.py --cameras frontleft,frontright
```

Run without OpenCV windows:

```bash
python examples/basic_streaming.py --no-display
```

Run in async mode:

```bash
python examples/basic_streaming.py --async-mode
```

Run two different streams on one robot:

```bash
python examples/basic_streaming.py --cameras frontleft,frontright --secondary-cameras left,right
```

Run the same stream configuration on two robots using the same username and password:

```bash
python examples/basic_streaming.py --robot-ip 192.168.80.3 --secondary-robot-ip 192.168.80.4 --username <user> --password <password> --cameras frontleft,frontright
```

Run two stream configurations on both robots:

```bash
python examples/basic_streaming.py --robot-ip 192.168.80.3 --secondary-robot-ip 192.168.80.4 --username <user> --password <password> --cameras frontleft,frontright --secondary-cameras left,right
```

Print timing information at the end:

```bash
python examples/basic_streaming.py --print-timing
```

Example with several options combined:

```bash
python examples/basic_streaming.py --async-mode --secondary-cameras left,right --duration 15 --no-display --print-timing
```

## Benchmark Example

`benchmark_allocation.py` compares allocation-heavy conversion against the in-place conversion path using a single captured image response set.

Show the full CLI:

```bash
python examples/benchmark_allocation.py --help
```

Run the benchmark:

```bash
python examples/benchmark_allocation.py --cameras frontleft,frontright --iters 100
```

## Notes

- Camera names are comma-separated values such as `frontleft`, `frontright`, `left`, `right`, `back`, and `hand`.
- `--secondary-cameras` adds a second stream configuration. If `--secondary-robot-ip` is also provided, both stream configurations are started on both robots.
- Press `q` in an OpenCV window to stop the streaming example early.
- If you do not want credentials stored in YAML, leave them blank in `config_example.yaml` and pass `--username` / `--password` when running the script.
