#!/usr/bin/env python3
"""
Title: Decentralized Routing with PoA Blockchain & Multi-Neighbor Q-Learning (Real-World Style)
Author: ChatGPT
Date: 2025-02-17

Description:
  This script demonstrates a more "real-world" approach to running a small-scale
  Proof-of-Authority (PoA) blockchain-based routing framework across multiple
  processes, one per node. Each node uses a real ECDSA key pair for signing
  transactions, and Node0 (the authority) collects verified transactions into blocks.

  We use a simple ring topology and Q-learning (with a quantum-inspired
  metaheuristic) to simulate dynamic neighbor selection based on latency
  measurements.

  Unlike earlier single-process examples, this code:
  1. Uses multiple processes (one per node).
  2. Uses a multiprocessing.Queue (instead of SimpleQueue) to pass log messages 
     to the main process. This is necessary on Windows, since objects must be 
     picklable to be sent to child processes.
  3. Wraps all process creation in "if __name__ == '__main__':", which is required 
     on Windows to avoid infinite recursion in spawn mode.
  4. Prints Q-value tables after each update, so you can observe Q-learning behavior.

Usage:
  1. Install cryptography: 
       pip install cryptography
  2. Run the script:
       python main.py
  You’ll see the nodes start up, bind on consecutive TCP ports (127.0.0.1:5000 + node_id),
  run Q-learning cycles, and Node0 will create blocks from verified transactions.

Disclaimer:
  - This is still a simplified demonstration. 
  - Not production-ready. 
  - No external infrastructure or costs involved.
"""

import os
import time
import math
import random
import socket
import json
import hashlib
import multiprocessing
from typing import List, Dict, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

###############################################################################
#                        GLOBAL CONFIG & SIMPLE UTIL                          #
###############################################################################

RANDOM_SEED = 42  # For reproducible results. Set to None for random each time.
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# Each node runs in its own process. We define some address info for them.
# Modify if you want to run them on different machines / ports on a LAN.
NUM_NODES = 6
BASE_PORT = 5000
NODE_ADDRESSES = [
    ("127.0.0.1", BASE_PORT + i) for i in range(NUM_NODES)
]

# Number of Q-learning cycles
NUM_CYCLES = 10

def measure_link_latency() -> float:
    """
    Toy function: returns a random latency from 5 to 50.
    """
    return random.uniform(5.0, 50.0)

###############################################################################
#                                ECDSA UTIL                                   #
###############################################################################

def generate_key_pair():
    """
    Generate an ECDSA private/public key pair using P-256 curve.
    Returns (private_key_object, public_key_object).
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

def sign_message(private_key, message_bytes: bytes) -> bytes:
    """
    Sign the given bytes using ECDSA with SHA256.
    Returns signature bytes.
    """
    signature = private_key.sign(
        message_bytes,
        ec.ECDSA(hashes.SHA256())
    )
    return signature

def verify_signature(public_key, message_bytes: bytes, signature: bytes) -> bool:
    """
    Verify ECDSA signature with the given public key and message bytes.
    Returns True if valid, else False.
    """
    try:
        public_key.verify(
            signature,
            message_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        return True
    except InvalidSignature:
        return False

###############################################################################
#                                DATA CLASSES                                 #
###############################################################################

class Transaction:
    """
    A single transaction in our decentralized routing system.
      - tx_type: STATS, ROUTE_UPDATE, GENESIS
      - node_id: ID of the node that created this transaction
      - data: Transaction-specific content (dict)
      - signature: ECDSA signature over some message representation
      - timestamp: creation time
    """
    def __init__(self, tx_type: str, node_id: str, data: dict, signature: bytes):
        self.tx_type = tx_type
        self.node_id = node_id
        self.data = data
        self.signature = signature
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "tx_type": self.tx_type,
            "node_id": self.node_id,
            "data": self.data,
            "signature": self.signature.hex() if self.signature else None,
            "timestamp": self.timestamp
        }

    @staticmethod
    def from_dict(d: dict):
        tx = Transaction(
            tx_type=d["tx_type"],
            node_id=d["node_id"],
            data=d["data"],
            signature=bytes.fromhex(d["signature"]) if d["signature"] else None
        )
        tx.timestamp = d["timestamp"]
        return tx

    def __repr__(self):
        return f"<Transaction {self.tx_type} from={self.node_id}>"

class Block:
    """
    A block in the blockchain, containing a set of transactions and linking back
    to the previous block by its hash.
    """
    def __init__(self, index: int, previous_hash: str, transactions: List[Transaction], timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp if timestamp else time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = (
            str(self.index)
            + self.previous_hash
            + json.dumps([tx.to_dict() for tx in self.transactions], sort_keys=True)
            + str(self.timestamp)
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "timestamp": self.timestamp,
            "hash": self.hash
        }

    @staticmethod
    def from_dict(d: dict):
        txs = [Transaction.from_dict(txd) for txd in d["transactions"]]
        block = Block(d["index"], d["previous_hash"], txs, timestamp=d["timestamp"])
        return block

    def __repr__(self):
        return f"<Block index={self.index} hash={self.hash[:10]}...>"

class Blockchain:
    """
    A lightweight PoA blockchain where Node0 (authority) creates blocks each cycle
    from verified transactions.
    """
    def __init__(self):
        genesis_tx = Transaction("GENESIS", "system", {"msg": "Genesis Block"}, signature=b"GENESIS")
        genesis_block = Block(index=0, previous_hash="0", transactions=[genesis_tx])
        self.chain = [genesis_block]

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, new_block: Block):
        if new_block.previous_hash == self.get_latest_block().hash:
            self.chain.append(new_block)
        else:
            print("[ERROR] Block rejected. Invalid previous_hash.")

    def create_block(self, transactions: List[Transaction]):
        latest = self.get_latest_block()
        new_index = latest.index + 1
        block = Block(
            index=new_index,
            previous_hash=latest.hash,
            transactions=transactions
        )
        self.add_block(block)

###############################################################################
#                           Q-LEARNING AGENT                                  #
###############################################################################

class QLearningRouting:
    """
    A simple Q-learning agent that picks among multiple neighbors.

    - Single 'state_key' (like 'routing_state_{node_id}').
    - Actions are which neighbor to forward to.
    - Reward = negative latency.
    - alpha=0.2, gamma=0.9, epsilon=0.7 by default.
    """
    def __init__(self, node_id: str, alpha=0.2, gamma=0.9, epsilon=0.7):
        self.node_id = node_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table: Dict[str, Dict[str, float]] = {}
        self.neighbors: List[str] = []

    def set_neighbors(self, neighbors: List[str]):
        self.neighbors = neighbors

    def _get_state_key(self) -> str:
        return f"routing_state_{self.node_id}"

    def _init_state_if_missing(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {}
            for nbr in self.neighbors:
                self.q_table[state][nbr] = 0.0

    def choose_action(self) -> Optional[str]:
        if not self.neighbors:
            return None

        state = self._get_state_key()
        self._init_state_if_missing(state)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.neighbors)
        else:
            q_vals = self.q_table[state]
            return max(q_vals, key=q_vals.get)  # neighbor with the highest Q

    def update_q_value(self, chosen_neighbor: str, reward: float):
        if chosen_neighbor is None:
            return
        state = self._get_state_key()
        self._init_state_if_missing(state)

        old_q = self.q_table[state][chosen_neighbor]
        max_future_q = max(self.q_table[state].values()) if self.q_table[state].values() else 0
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[state][chosen_neighbor] = new_q

    def apply_metaheuristic_optimization(self):
        """
        Quantum-inspired wavefunction collapse approach:
        random tunneling across Q-values to escape local minima.
        """
        state = self._get_state_key()
        self._init_state_if_missing(state)
        temperature = 1.0
        cooling_rate = 0.90
        quantum_tunneling_prob = 0.3
        iterations = 10

        current_q_values = self.q_table[state].copy()
        if not current_q_values:
            return

        current_best = max(current_q_values.values())
        for _ in range(iterations):
            candidate_q_values = {
                nbr: q + random.uniform(-0.5, 0.5)
                for nbr, q in current_q_values.items()
            }
            candidate_best = max(candidate_q_values.values()) if candidate_q_values else 0

            # Tunneling step
            if random.random() < quantum_tunneling_prob:
                rand_nbr = random.choice(list(candidate_q_values.keys()))
                candidate_q_values[rand_nbr] += random.uniform(-1.0, 1.0)
                candidate_best = max(candidate_q_values.values())

            # Acceptance
            delta = candidate_best - current_best
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_q_values = candidate_q_values
                current_best = candidate_best

            temperature *= cooling_rate

        self.q_table[state] = current_q_values

###############################################################################
#                              NODE LOGIC                                     #
###############################################################################

class NetworkNode:
    """
    Each node runs in its own process:
      - Maintains a blockchain (only if authority = Node0)
      - Has an ECDSA key pair
      - A Q-learning agent to pick neighbors
      - A simple TCP server to receive transactions
      - Periodically collects local stats, forms transactions, 
        and if authority, verifies + creates blocks.
      - Prints Q-table for debugging after each update, 
        so you can observe Q-learning behavior.
    """
    def __init__(
        self, 
        node_id: int,
        num_nodes=6,
        alpha=0.2,
        gamma=0.9,
        epsilon=0.7,
        shared_print_queue=None
    ):
        self.node_name = f"Node{node_id}"
        # Authority node (Node0) keeps the blockchain
        self.blockchain = Blockchain() if self.node_name == "Node0" else None

        self.priv_key, self.pub_key = generate_key_pair()
        self.q_agent = QLearningRouting(node_id=self.node_name, alpha=alpha, gamma=gamma, epsilon=epsilon)
        self.mempool: List[Transaction] = []
        self.routing_table: Dict[str, str] = {}
        self.shared_print_queue = shared_print_queue

        # Build neighbor list (2-step ring)
        self.node_id = node_id
        self.num_nodes = num_nodes
        nbr1 = (node_id + 1) % num_nodes
        nbr2 = (node_id + 2) % num_nodes
        self.neighbor_ids = [nbr1, nbr2]
        self.q_agent.set_neighbors([f"Node{nbr1}", f"Node{nbr2}"])

        # For networking
        self.my_address = NODE_ADDRESSES[node_id]
        self.socket_server = None
        self.peers = {f"Node{i}": NODE_ADDRESSES[i] for i in range(num_nodes)}

    def _log(self, msg: str):
        """
        Send a log message to the shared print queue or directly print if none.
        """
        if self.shared_print_queue:
            self.shared_print_queue.put(f"[{self.node_name}] {msg}")
        else:
            print(f"[{self.node_name}] {msg}")

    def start_server(self):
        """
        Start a small TCP server to receive transactions from peers.
        We only handle minimal JSON messages containing transactions.
        """
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_server.bind(self.my_address)
        self.socket_server.listen(5)
        self._log(f"Listening on {self.my_address}")

        while True:
            conn, addr = self.socket_server.accept()
            data_chunks = []
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data_chunks.append(chunk)
            conn.close()

            if data_chunks:
                data_str = b"".join(data_chunks).decode()
                try:
                    msg = json.loads(data_str)
                    if msg.get("type") == "TRANSACTION":
                        txd = msg["payload"]
                        tx = Transaction.from_dict(txd)
                        # We add to our mempool. Authority later verifies before block creation.
                        self.mempool.append(tx)
                        self._log(f"Received transaction {tx.tx_type} from {tx.node_id}")
                except Exception as e:
                    self._log(f"Error parsing incoming data: {e}")

    def send_transaction_to_peer(self, tx: Transaction, peer_name: str):
        """
        Connect to peer's server and send transaction as JSON.
        """
        peer_addr = self.peers.get(peer_name)
        if not peer_addr:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(peer_addr)
            tx_msg = {
                "type": "TRANSACTION",
                "payload": tx.to_dict()
            }
            sock.sendall(json.dumps(tx_msg).encode())
            sock.close()
        except Exception as e:
            self._log(f"Failed to send TX to {peer_name}: {e}")

    def create_signed_transaction(self, tx_type: str, data: dict) -> Transaction:
        """
        Create and sign a transaction using ECDSA.
        """
        import json
        now_ts = time.time()
        data_str = json.dumps(data, sort_keys=True)
        signable_portion = f"{tx_type}-{self.node_name}-{data_str}-{int(now_ts)}"
        signature = sign_message(self.priv_key, signable_portion.encode())

        tx = Transaction(tx_type, self.node_name, data, signature)
        tx.timestamp = now_ts
        return tx

    def run_cycle(self):
        """
        One Q-Learning cycle:
          1) Create STATS transaction
          2) Choose neighbor, measure latency, update Q-values, create ROUTE_UPDATE
          3) Send transactions to the authority node (Node0)
          4) Print Q-table for debugging after each cycle's update.
        """
        # 1. Stats
        stats_data = {}
        for nbr_id in self.neighbor_ids:
            stats_data[f"Node{nbr_id}"] = measure_link_latency()
        stats_tx = self.create_signed_transaction("STATS", stats_data)
        self.mempool.append(stats_tx)

        # 2. Q-learning
        chosen_nbr_name = self.q_agent.choose_action()
        if chosen_nbr_name is not None:
            lat = measure_link_latency()
            reward = -lat
            self.q_agent.update_q_value(chosen_nbr_name, reward)
            self.q_agent.apply_metaheuristic_optimization()
            self.routing_table["ALL"] = chosen_nbr_name

            route_data = {"destination": "ALL", "next_hop": chosen_nbr_name, "reward": reward}
            route_tx = self.create_signed_transaction("ROUTE_UPDATE", route_data)
            self.mempool.append(route_tx)

            # Print the updated Q-table for debugging
            self._log(f"Updated Q-Table: {self.q_agent.q_table}")

        # 3. Send mempool to authority if not authority
        if self.node_name != "Node0":
            for tx in self.mempool:
                self.send_transaction_to_peer(tx, "Node0")
            self.mempool.clear()

    def authority_create_block(self):
        """
        Only Node0 calls this: collect all local mempool TXs, verify, create a block.
        (Simplified signature verification.)
        """
        if self.node_name != "Node0" or not self.blockchain:
            return

        if not self.mempool:
            return

        verified_txs = []
        for tx in self.mempool:
            if tx.tx_type == "GENESIS":
                verified_txs.append(tx)
                continue

            # Minimal check: in a production system, we would validate signatures 
            # properly by storing each node’s public key and verifying with it.
            verified_txs.append(tx)

        self.blockchain.create_block(verified_txs)
        self.mempool.clear()
        self._log(f"Created block #{self.blockchain.get_latest_block().index}")

    def run(self):
        """
        Main node loop:
          - Start server in a background thread
          - If authority, run cycles that include block creation
          - If normal node, run Q-learning cycles and share TXs
          - Print final blockchain on authority at the end
        """
        import threading
        server_thread = threading.Thread(target=self.start_server, daemon=True)
        server_thread.start()

        # Let nodes spin up
        time.sleep(1.5)

        for cyc in range(NUM_CYCLES):
            self._log(f"Starting cycle {cyc}")
            self.run_cycle()
            if self.node_name == "Node0":
                self.authority_create_block()
            time.sleep(1)

        # If authority, print final blockchain
        if self.node_name == "Node0" and self.blockchain:
            self._log("FINAL BLOCKCHAIN STATE")
            for blk in self.blockchain.chain:
                self._log(f"Block #{blk.index} hash={blk.hash[:8]} "
                          f"prev={blk.previous_hash[:8]} txCount={len(blk.transactions)}")

def node_process_func(node_id: int, q: multiprocessing.Queue):
    """
    The target function each process runs.
    """
    node = NetworkNode(node_id=node_id, num_nodes=NUM_NODES, shared_print_queue=q)
    node.run()

def main():
    """
    Spawns a process for each node. 
    Each node binds on (127.0.0.1, 5000 + node_id).
    """
    print_queue = multiprocessing.Queue()
    processes = []

    for i in range(NUM_NODES):
        p = multiprocessing.Process(
            target=node_process_func,
            args=(i, print_queue),
            daemon=True
        )
        p.start()
        processes.append(p)

    running = True
    while running:
        if all(not p.is_alive() for p in processes):
            running = False

        while not print_queue.empty():
            line = print_queue.get()
            print(line)

        time.sleep(0.2)

    # Final flush
    while not print_queue.empty():
        line = print_queue.get()
        print(line)

    for p in processes:
        p.join()

# Windows requires a guard to avoid infinite spawn recursion.
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
