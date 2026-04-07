import re
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx


class NodeType(Enum):
    START = "START"
    END = "END"
    PULLER_RB = "PULLER_RB"
    PULLER_YG = "PULLER_YG"
    PAINTER_RED = "PAINTER_RED"
    PAINTER_BLUE = "PAINTER_BLUE"
    PAINTER_YELLOW = "PAINTER_YELLOW"
    PAINTER_GREEN = "PAINTER_GREEN"


@dataclass
class Route:
    condition: str | None  # For pullers: 'R', 'B', 'Y', 'G', 'EMPTY'. For painters: None
    target: str  # Node ID or 'NONE' or 'END'


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    routes: list[Route]
    line_number: int


@dataclass
class Robot:
    tape: str  # Sequence of colors like "RBR" or ""

    def pull_front(self) -> str | None:
        """Remove and return the first color from tape, or None if empty."""
        if not self.tape:
            return None
        color = self.tape[0]
        self.tape = self.tape[1:]
        return color

    def paint_back(self, color: str):
        """Add a color to the end of the tape."""
        self.tape += color

    def is_empty(self) -> bool:
        """Check if tape is empty."""
        return len(self.tape) == 0

    def peek_front(self) -> str | None:
        """Look at the first color without removing it."""
        return self.tape[0] if self.tape else None


@dataclass
class ExecutionResult:
    finished: bool
    final_tape: str
    path: list[str]  # Node IDs visited during execution
    rejection_reason: str | None = None


@dataclass
class JudgementResult:
    verdict: str  # "ACCEPT" or "REJECT"
    execution_result: ExecutionResult
    criteria_matched: bool
    reason: str  # Explanation for the verdict


class RobotFactory:
    """A factory that can process robots according to the parsed DSL."""

    def __init__(self, nodes: dict[str, Node]):
        self.nodes = nodes
        self.start_node = "start"
        self.end_node = "end"

        # Validate that we have required nodes
        if self.start_node not in nodes:
            raise ValueError("Factory must have a START node with ID 'start'")
        if self.end_node not in nodes:
            raise ValueError("Factory must have an END node with ID 'end'")

    def process_robot(self, input_tape: str) -> ExecutionResult:
        """Process a robot through the factory and return the result."""
        robot = Robot(input_tape)
        path = []
        current_node_id = self.start_node

        # Maximum iterations to prevent infinite loops
        max_iterations = 1000
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            path.append(current_node_id)

            # Check if we've reached a terminal state
            if current_node_id == "NONE":
                return ExecutionResult(
                    finished=False, final_tape=robot.tape, path=path, rejection_reason="Routed to NONE"
                )

            if current_node_id == self.end_node:
                return ExecutionResult(finished=True, final_tape=robot.tape, path=path)

            # Get the current node
            if current_node_id not in self.nodes:
                return ExecutionResult(
                    finished=False,
                    final_tape=robot.tape,
                    path=path,
                    rejection_reason=f"Unknown node: {current_node_id}",
                )

            current_node = self.nodes[current_node_id]

            # Process the node based on its type
            try:
                next_node_id = self._process_node(robot, current_node)
                current_node_id = next_node_id
            except Exception as e:
                return ExecutionResult(
                    finished=False,
                    final_tape=robot.tape,
                    path=path,
                    rejection_reason=f"Error processing node {current_node_id}: {str(e)}",
                )

        # If we've exceeded max iterations, assume infinite loop
        return ExecutionResult(
            finished=False,
            final_tape=robot.tape,
            path=path,
            rejection_reason="Maximum iterations exceeded (possible infinite loop)",
        )

    def _process_node(self, robot: Robot, node: Node) -> str:
        """Process a robot through a single node and return the next node ID."""

        if node.node_type == NodeType.START:
            # START node: just route to next
            return node.routes[0].target

        elif node.node_type == NodeType.END:
            # END node: should not be processed (terminal)
            return "END"

        elif node.node_type in [NodeType.PULLER_RB, NodeType.PULLER_YG]:
            # Puller node: only remove color if this puller type can handle it
            front_color = robot.peek_front()

            # Check if tape is empty or if puller can't handle the front color
            if front_color is None:
                # Tape is actually empty
                for route in node.routes:
                    if route.condition == "EMPTY":
                        return route.target
                return "NONE"

            # Check if this puller can handle the front color
            if node.node_type == NodeType.PULLER_RB:
                can_handle = front_color in ["R", "B"]
            else:  # NodeType.PULLER_YG
                can_handle = front_color in ["Y", "G"]

            if can_handle:
                # Remove the color and route based on it
                removed_color = robot.pull_front()
                for route in node.routes:
                    if route.condition == removed_color:
                        return route.target
                # No matching route for this color
                return "NONE"
            else:
                # Cannot handle this color, route to EMPTY branch without removing color
                for route in node.routes:
                    if route.condition == "EMPTY":
                        return route.target
                return "NONE"

        elif node.node_type.name.startswith("PAINTER_"):
            # Painter node: add color to back and route to next
            color_map = {
                NodeType.PAINTER_RED: "R",
                NodeType.PAINTER_BLUE: "B",
                NodeType.PAINTER_YELLOW: "Y",
                NodeType.PAINTER_GREEN: "G",
            }
            color = color_map[node.node_type]
            robot.paint_back(color)
            return node.routes[0].target

        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

    def test_multiple_inputs(self, test_cases: list[str]) -> list[ExecutionResult]:
        """Test multiple input tapes and return results."""
        results = []
        for input_tape in test_cases:
            result = self.process_robot(input_tape)
            results.append(result)
        return results

    def evaluate_robot(self, input_tape: str, output_criteria: str) -> JudgementResult:
        """
        Evaluate a robot's execution against output criteria.

        Args:
            input_tape: The input tape for the robot
            output_criteria: Regex pattern that the output tape must match

        Returns:
            JudgementResult with verdict (ACCEPT/REJECT), execution details, and reasoning
        """
        # Process the robot through the factory
        execution_result = self.process_robot(input_tape)

        # Check if execution finished successfully
        if not execution_result.finished:
            return JudgementResult(
                verdict="REJECT",
                execution_result=execution_result,
                criteria_matched=False,
                reason=f"Execution did not finish: {execution_result.rejection_reason}",
            )

        # Check if output matches criteria
        try:
            criteria_matched = bool(re.match(output_criteria, execution_result.final_tape))
        except re.error as e:
            return JudgementResult(
                verdict="REJECT",
                execution_result=execution_result,
                criteria_matched=False,
                reason=f"Invalid regex pattern: {str(e)}",
            )

        if criteria_matched:
            return JudgementResult(
                verdict="ACCEPT",
                execution_result=execution_result,
                criteria_matched=True,
                reason="Execution finished and output matches criteria",
            )
        else:
            return JudgementResult(
                verdict="REJECT",
                execution_result=execution_result,
                criteria_matched=False,
                reason=f"Output '{execution_result.final_tape}' does not match criteria '{output_criteria}'",
            )

    def print_execution_trace(self, input_tape: str):
        """Print a detailed trace of robot execution."""
        print(f"\nProcessing robot with input tape: '{input_tape}'")
        print("=" * 50)

        result = self.process_robot(input_tape)

        # Show path
        print("Execution path:")
        for i, node_id in enumerate(result.path):
            if i < len(result.path) - 1:
                print(f"  {i + 1}. {node_id} -> {result.path[i + 1]}")
            else:
                print(f"  {i + 1}. {node_id} (final)")

        print("\nFinal result:")
        print(f"  Input tape:  '{input_tape}'")
        print(f"  Output tape: '{result.final_tape}'")
        print(f"  Finished:    {result.finished}")
        if result.rejection_reason:
            print(f"  Rejection:   {result.rejection_reason}")

    def evaluate_multiple_inputs(self, test_cases: list[str], output_criteria: str) -> list[JudgementResult]:
        """
        Evaluate multiple input tapes against the same output criteria.

        Args:
            test_cases: List of input tapes to test
            output_criteria: Regex pattern that outputs must match

        Returns:
            List of JudgementResult objects
        """
        results = []
        for input_tape in test_cases:
            result = self.evaluate_robot(input_tape, output_criteria)
            results.append(result)
        return results


class ParseError(Exception):
    def __init__(self, message: str, line_number: int = None):
        self.line_number = line_number
        if line_number:
            super().__init__(f"Line {line_number}: {message}")
        else:
            super().__init__(message)


class ManufactoriaParser:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.start_node: str | None = None
        self.end_node: str | None = None

    def parse(self, text: str) -> dict[str, Node]:
        """Parse the DSL text and return a dictionary of nodes."""
        self.nodes = {}
        self.start_node = None
        self.end_node = None

        lines = text.strip().split("\n")

        # Remove code block markers if present
        if lines and lines[0].strip() == "```":
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        # Remove empty lines and comments
        processed_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                processed_lines.append((line, i + 1))

        if not processed_lines:
            raise ParseError("Empty program")

        i = 0
        while i < len(processed_lines):
            line, line_num = processed_lines[i]
            i = self._parse_node(processed_lines, i)

        self._validate_program()
        return self.nodes

    def parse_and_create_factory(self, text: str) -> RobotFactory:
        """Parse DSL text and return a RobotFactory object."""
        nodes = self.parse(text)
        return RobotFactory(nodes)

    def _parse_node(self, lines: list[tuple[str, int]], start_idx: int) -> int:
        """Parse a single node starting at start_idx. Returns the next index to process."""
        line, line_num = lines[start_idx]

        # Match node declaration (now allowing trailing comments)
        node_match = re.match(
            r"^(START|END|PULLER_RB|PULLER_YG|PAINTER_RED|PAINTER_BLUE|PAINTER_YELLOW|PAINTER_GREEN)\s+(\w+):?\s*(?:#.*)?$",
            line.strip(),
        )

        if not node_match:
            raise ParseError(f"Invalid node declaration: {line.strip()}", line_num)

        node_type_str, node_id = node_match.groups()
        node_type = NodeType(node_type_str)

        # Validate colon usage: END nodes should not have colons, other nodes should have colons
        has_colon = ":" in line.split("#")[0]  # Check for colon before any comment
        if node_type == NodeType.END and has_colon:
            raise ParseError("END nodes should not have a colon", line_num)
        elif node_type != NodeType.END and not has_colon:
            raise ParseError(f"{node_type_str} nodes must have a colon", line_num)

        # Check for duplicate node IDs
        if node_id in self.nodes:
            raise ParseError(f"Duplicate node ID: {node_id}", line_num)

        # Special handling for START and END
        if node_type == NodeType.START:
            if node_id != "start":
                raise ParseError("START node must have ID 'start'", line_num)
            if self.start_node is not None:
                raise ParseError("Multiple START nodes found", line_num)
            self.start_node = node_id
        elif node_type == NodeType.END:
            if node_id != "end":
                raise ParseError("END node must have ID 'end'", line_num)
            if self.end_node is not None:
                raise ParseError("Multiple END nodes found", line_num)
            self.end_node = node_id

        # Parse routes
        routes = []
        current_idx = start_idx + 1

        while current_idx < len(lines):
            route_line, route_line_num = lines[current_idx]

            # Check if this line belongs to the current node (indented)
            if not route_line.startswith("    ") and not route_line.startswith("\t"):
                break

            route = self._parse_route(route_line, route_line_num, node_type)
            routes.append(route)
            current_idx += 1

        # Validate routes based on node type
        self._validate_routes(node_type, routes, line_num)

        self.nodes[node_id] = Node(node_id, node_type, routes, line_num)
        return current_idx

    def _parse_route(self, line: str, line_num: int, node_type: NodeType) -> Route:
        """Parse a single route line."""
        stripped = line.strip()

        if node_type in [NodeType.PULLER_RB, NodeType.PULLER_YG]:
            # Puller route: [COLOR] target or [EMPTY] target (now allowing trailing comments)
            match = re.match(r"^\[([RBYG]|EMPTY)\]\s+(\w+|NONE|END)\s*(?:#.*)?$", stripped)
            if not match:
                raise ParseError(f"Invalid puller route format: {stripped}", line_num)
            condition, target = match.groups()
            return Route(condition, target)

        elif node_type.name.startswith("PAINTER_") or node_type == NodeType.START:
            # Painter/Start route: NEXT target (now allowing trailing comments)
            match = re.match(r"^NEXT\s+(\w+|NONE|END)\s*(?:#.*)?$", stripped)
            if not match:
                raise ParseError(f"Invalid NEXT route format: {stripped}", line_num)
            target = match.group(1)
            return Route(None, target)

        elif node_type == NodeType.END:
            raise ParseError("END nodes cannot have routes", line_num)

        else:
            raise ParseError(f"Unknown node type for routing: {node_type}", line_num)

    def _validate_routes(self, node_type: NodeType, routes: list[Route], line_num: int):
        """Validate that routes are appropriate for the node type."""
        if node_type == NodeType.START:
            if len(routes) != 1:
                raise ParseError("START node must have exactly one NEXT route", line_num)

        elif node_type == NodeType.END:
            if len(routes) != 0:
                raise ParseError("END node cannot have routes", line_num)

        elif node_type == NodeType.PULLER_RB:
            valid_conditions = {"R", "B", "EMPTY"}
            conditions = {route.condition for route in routes}
            for condition in conditions:
                if condition not in valid_conditions:
                    raise ParseError(f"Invalid condition '{condition}' for PULLER_RB", line_num)

        elif node_type == NodeType.PULLER_YG:
            valid_conditions = {"Y", "G", "EMPTY"}
            conditions = {route.condition for route in routes}
            for condition in conditions:
                if condition not in valid_conditions:
                    raise ParseError(f"Invalid condition '{condition}' for PULLER_YG", line_num)

        elif node_type.name.startswith("PAINTER_") and len(routes) != 1:
            raise ParseError("PAINTER nodes must have exactly one NEXT route", line_num)

    def _validate_program(self):
        """Validate the overall program structure."""
        if self.start_node is None:
            raise ParseError("Program must have a START node")

        if self.end_node is None:
            raise ParseError("Program must have an END node")

        # Check that all referenced nodes exist
        for node in self.nodes.values():
            for route in node.routes:
                target = route.target
                if target not in ["NONE", "END"] and target not in self.nodes:
                    raise ParseError(f"Node '{node.node_id}' references undefined node '{target}'", node.line_number)

    def visualize(self, nodes: dict[str, Node], save_path: str | None = None, show: bool = True):
        """Create a visual representation of the factory network."""
        G = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in nodes.items():
            color = self._get_node_color(node.node_type)
            G.add_node(node_id, color=color, type=node.node_type.value)

        # Add edges
        for node_id, node in nodes.items():
            for route in node.routes:
                target = route.target
                if target == "NONE":
                    # Add a special NONE node if it doesn't exist
                    if "NONE" not in G:
                        G.add_node("NONE", color="red", type="NONE")
                    edge_label = route.condition if route.condition else "NEXT"
                    G.add_edge(node_id, "NONE", label=edge_label)
                elif target == "END":
                    edge_label = route.condition if route.condition else "NEXT"
                    G.add_edge(node_id, "end", label=edge_label)
                else:
                    edge_label = route.condition if route.condition else "NEXT"
                    G.add_edge(node_id, target, label=edge_label)

        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Draw nodes
        for node_id in G.nodes():
            color = G.nodes[node_id]["color"]
            nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_color=color, node_size=1500, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->", width=2)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        plt.title("Manufactoria Factory Network", fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    def _get_node_color(self, node_type: NodeType) -> str:
        """Get color for node visualization."""
        color_map = {
            NodeType.START: "lightgreen",
            NodeType.END: "lightcoral",
            NodeType.PULLER_RB: "lightblue",
            NodeType.PULLER_YG: "lightyellow",
            NodeType.PAINTER_RED: "pink",
            NodeType.PAINTER_BLUE: "lightsteelblue",
            NodeType.PAINTER_YELLOW: "khaki",
            NodeType.PAINTER_GREEN: "lightseagreen",
        }
        return color_map.get(node_type, "lightgray")


def parse_manufactoria_dsl(text: str) -> dict[str, Node]:
    """Convenience function to parse DSL text."""
    parser = ManufactoriaParser()
    return parser.parse(text)


def create_robot_factory(text: str) -> RobotFactory:
    """Convenience function to parse DSL text and create a RobotFactory."""
    parser = ManufactoriaParser()
    return parser.parse_and_create_factory(text)


def visualize_manufactoria_factory(text: str, save_path: str | None = None, show: bool = True):
    """Convenience function to parse and visualize DSL text."""
    parser = ManufactoriaParser()
    nodes = parser.parse(text)
    parser.visualize(nodes, save_path, show)
    return nodes


# Example usage and testing
if __name__ == "__main__":
    # Test with the example from the DSL documentation
    example_dsl = """
    START start:
        NEXT entry

    PULLER_RB entry:
        [R] end

    END end
    """

    try:
        # Parse and create factory
        factory = create_robot_factory(example_dsl)

        print("Parsing successful!")
        print(f"Factory created with {len(factory.nodes)} nodes")

        # Test various inputs
        test_cases = ["R", "B", "RR", "BR", ""]
        print(f"\nTesting factory with inputs: {test_cases}")

        for input_tape in test_cases:
            result = factory.process_robot(input_tape)
            print(f"Input: '{input_tape}' -> Output: '{result.final_tape}', Finished: {result.finished}")

        # Show detailed trace for one example
        factory.print_execution_trace("R")

        # Test evaluation with output criteria
        print("\nTesting evaluation with output criteria:")
        criteria = "R"  # Output must be exactly "R"

        for input_tape in test_cases:
            judgement = factory.evaluate_robot(input_tape, criteria)
            print(f"Input: '{input_tape}' -> Verdict: {judgement.verdict}, Reason: {judgement.reason}")

    except ParseError as e:
        print(f"Parse error: {e}")
