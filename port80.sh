#!/bin/sh
# Redirect port 80 -> 8888 for bird watcher web dashboard
#
# Two rules needed:
#   PREROUTING  — catches external traffic arriving on the Pi's IP
#   OUTPUT -o lo — catches local traffic (e.g. Dataplicity wormhole agent)

RULE_PRE="-t nat -p tcp --dport 80 -j REDIRECT --to-port 8888"
RULE_OUT="-t nat -o lo -p tcp --dport 80 -j REDIRECT --to-port 8888"

add_rules() {
  sudo iptables -t nat -C PREROUTING $RULE_PRE 2>/dev/null || sudo iptables -t nat -A PREROUTING $RULE_PRE
  sudo iptables -t nat -C OUTPUT $RULE_OUT 2>/dev/null || sudo iptables -t nat -A OUTPUT $RULE_OUT
}

del_rules() {
  sudo iptables -t nat -D PREROUTING $RULE_PRE 2>/dev/null
  sudo iptables -t nat -D OUTPUT $RULE_OUT 2>/dev/null
}

check_rules() {
  sudo iptables -t nat -C PREROUTING $RULE_PRE 2>/dev/null && PRE=1 || PRE=0
  sudo iptables -t nat -C OUTPUT $RULE_OUT 2>/dev/null && OUT=1 || OUT=0
}

case "$1" in
  on)
    add_rules
    echo "Aktivert: port 80 -> 8888 (ekstern + lokal/Dataplicity)"
    ;;
  off)
    del_rules
    echo "Deaktivert: port 80 -> 8888"
    ;;
  status)
    check_rules
    if [ $PRE -eq 1 ] && [ $OUT -eq 1 ]; then
      echo "Aktiv: port 80 -> 8888 (ekstern + lokal)"
    elif [ $PRE -eq 1 ]; then
      echo "Delvis aktiv: kun ekstern (mangler lokal/Dataplicity-regel)"
    elif [ $OUT -eq 1 ]; then
      echo "Delvis aktiv: kun lokal (mangler ekstern-regel)"
    else
      echo "Ikke aktiv"
    fi
    ;;
  *)
    echo "Bruk: $0 {on|off|status}"
    exit 1
    ;;
esac
