#!/bin/bash
# Runs inside a ROS 2 container: real node, real topics and services, real CLI tools.
#   docker run --rm -v "$PWD:/ws" -w /ws ros:jazzy-ros-base bash tests/ros/verify.sh
source /opt/ros/*/setup.bash
set -u
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq python3-pip >/dev/null 2>&1
pip3 install -q --break-system-packages "pydantic>=2.6" pyyaml numpy 2>&1 | tail -1
export PYTHONPATH="$(printf '%s:' /ws/packages/*/src)$PYTHONPATH"
LOG=$(mktemp -d)
fails=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
field() { python3 -c "
import json, sys
text = sys.stdin.read()
start = text.index('data: ') + 6
line = text[start:].split('\n')[0].strip()
if line[0] in '\"\'': line = line[1:-1].replace(\"''\", \"'\")
d = json.loads(line)
print(eval(sys.argv[1], {'d': d}))" "$1"; }

python3 -m affect_ros.node --profile quadruped --personality pet --log-dir "$LOG" >"$LOG/node.out" 2>&1 &
NODE=$!
sleep 4

check "node is running" "kill -0 $NODE" || true
kill -0 $NODE 2>/dev/null || { echo "node output:"; cat "$LOG/node.out"; exit 1; }
check "topics exist" "ros2 topic list | grep -q /affect/command && ros2 topic list | grep -q /affect/state"
check "services exist" "ros2 service list | grep -q /affect/stop"

HZ=$(timeout 8 ros2 topic hz /affect/state --window 30 2>/dev/null | grep -m1 'average rate' | awk '{print $3}')
echo "     state rate: $HZ Hz"
check "state publishes near 30 Hz" "python3 -c 'import sys; sys.exit(0 if 20 < float(\"${HZ:-0}\") < 40 else 1)'"

BASE=$(timeout 5 ros2 topic echo --full-length --once /affect/command std_msgs/msg/String 2>/dev/null | field "d['channels']['body.height']")
echo "     resting body.height: $BASE"
timeout 12 ros2 topic echo --full-length /affect/command std_msgs/msg/String >"$LOG/commands.txt" 2>/dev/null &
sleep 2

THREAT='{"cause":"a threat","source":"ros","relevance":1,"congruence":-0.8,"novelty":0.8,"agency":"user","control":-0.9,"norm":-0.3}'
ros2 topic pub --once -w 1 /affect/appraisal std_msgs/msg/String "{data: '$THREAT'}" >/dev/null 2>&1
sleep 1
wait %2 2>/dev/null
sleep 6
LOW=$(python3 -c "
import json, re
values = []
for line in open('$LOG/commands.txt'):
    match = re.search(r'body.height.{0,6}: ?([0-9.]+)', line)
    if match: values.append(float(match.group(1)))
print(min(values))")
read -r LABELS CAUSE < <(python3 -c "
import json
labels, cause = [], ''
for line in open('$LOG/ros-session.jsonl'):
    try: r = json.loads(line)
    except ValueError: continue
    if r['kind'] == 'state':
        if not labels or labels[-1] != r['data']['label']: labels.append(r['data']['label'])
        if r['data']['label'] == 'fear': cause = r['data']['cause']
print(','.join(labels), cause.replace(' ', '_'))")
echo "     after threat: labels=$LABELS cause=$CAUSE lowest body.height=$LOW"
check "appraisal changes the emotion" "echo '$LABELS' | grep -q fear"
check "cause is carried" "[ '$CAUSE' = 'a_threat' ]"
check "the body crouches" "python3 -c 'import sys; sys.exit(0 if $LOW < $BASE - 0.02 else 1)'"

ros2 topic pub --once -w 1 /affect/appraisal std_msgs/msg/String "{data: 'not json'}" >/dev/null 2>&1
ros2 topic pub --once -w 1 /affect/appraisal std_msgs/msg/String "{data: '{\"cause\":\"x\",\"relevance\":99,\"congruence\":0,\"novelty\":0,\"control\":0,\"norm\":0}'}" >/dev/null 2>&1
sleep 1
check "bad input is rejected and logged" "[ \$(grep -c 'rejected appraisal' '$LOG/node.out') -ge 2 ]"
check "node survived bad input" "kill -0 $NODE"

ros2 service call /affect/stop std_srvs/srv/Trigger >/dev/null 2>&1
sleep 0.5
QUIET=$(timeout 2 ros2 topic echo --full-length --once /affect/command std_msgs/msg/String 2>/dev/null | grep -c data)
check "stop silences commands" "[ '$QUIET' = 0 ]"
ros2 service call /affect/resume std_srvs/srv/Trigger >/dev/null 2>&1
sleep 0.5
BACK=$(timeout 5 ros2 topic echo --full-length --once /affect/command std_msgs/msg/String 2>/dev/null | grep -c data)
check "resume restores commands" "[ '$BACK' -ge 1 ]"

kill -INT $NODE; wait $NODE 2>/dev/null
REPLAY=$(python3 -m affect_core replay "$LOG/ros-session.jsonl")
echo "     $REPLAY"
check "a live ROS session replays exactly" "echo '$REPLAY' | grep -q '^exact:'"

echo "fails=$fails"
exit $fails
