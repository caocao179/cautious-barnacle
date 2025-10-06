"""
DC Optimal Power Flow (DC-OPF) Module for Industrial Electrical Equipment Analysis

This module implements DC-OPF calculations for carbon footprint optimization
and power flow analysis in industrial settings. Python 3.13 compatible.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
from enum import Enum
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import cvxpy as cp


class BusType(Enum):
    """Enumeration for bus types in power system."""
    SLACK = "slack"
    PV = "pv"
    PQ = "pq"


@dataclass
class Bus:
    """Data class representing a bus in the power system."""
    id: int
    name: str
    bus_type: BusType
    voltage_magnitude: float = 1.0  # per unit
    voltage_angle: float = 0.0  # radians
    real_power_demand: float = 0.0  # MW
    reactive_power_demand: float = 0.0  # MVAr
    real_power_generation: float = 0.0  # MW
    reactive_power_generation: float = 0.0  # MVAr
    voltage_min: float = 0.95  # per unit
    voltage_max: float = 1.05  # per unit


@dataclass
class Branch:
    """Data class representing a branch (transmission line) in the power system."""
    id: int
    from_bus: int
    to_bus: int
    resistance: float  # per unit
    reactance: float  # per unit
    susceptance: float = 0.0  # per unit
    tap_ratio: float = 1.0
    phase_shift: float = 0.0  # radians
    power_rating: float = float('inf')  # MVA
    status: bool = True


@dataclass
class Generator:
    """Data class representing a generator in the power system."""
    id: int
    bus_id: int
    real_power_min: float  # MW
    real_power_max: float  # MW
    reactive_power_min: float  # MVAr
    reactive_power_max: float  # MVAr
    cost_coefficient_a: float = 0.0  # $/MW^2
    cost_coefficient_b: float = 10.0  # $/MW
    cost_coefficient_c: float = 0.0  # $
    carbon_emission_rate: float = 0.5  # kg CO2/MWh
    ramp_rate_up: float = float('inf')  # MW/h
    ramp_rate_down: float = float('inf')  # MW/h
    status: bool = True


class DCOptimalPowerFlow:
    """
    DC Optimal Power Flow solver for industrial electrical equipment analysis.
    
    This implementation uses linearized power flow equations and convex optimization
    to minimize carbon footprint and operational costs.
    """
    
    def __init__(self, base_mva: float = 100.0):
        """
        Initialize DC-OPF solver.
        
        Args:
            base_mva: Base MVA for per-unit calculations
        """
        self.base_mva = base_mva
        self.buses: Dict[int, Bus] = {}
        self.branches: Dict[int, Branch] = {}
        self.generators: Dict[int, Generator] = {}
        
        # System matrices
        self.admittance_matrix = None
        self.ptdf_matrix = None  # Power Transfer Distribution Factor matrix
        
        # Solution
        self.voltage_angles = None
        self.power_flows = None
        self.generator_dispatch = None
        self.total_cost = None
        self.total_emissions = None
        
    def add_bus(self, bus: Bus) -> None:
        """Add a bus to the power system."""
        self.buses[bus.id] = bus
        
    def add_branch(self, branch: Branch) -> None:
        """Add a branch to the power system."""
        self.branches[branch.id] = branch
        
    def add_generator(self, generator: Generator) -> None:
        """Add a generator to the power system."""
        self.generators[generator.id] = generator
        
    def build_admittance_matrix(self) -> np.ndarray:
        """
        Build the bus admittance matrix for DC power flow.
        
        Returns:
            Bus admittance matrix (reactance part only for DC)
        """
        n_buses = len(self.buses)
        if n_buses == 0:
            raise ValueError("No buses defined in the system")
        
        bus_ids = sorted(self.buses.keys())
        bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        
        # Initialize admittance matrix
        Y = np.zeros((n_buses, n_buses), dtype=complex)
        
        # Add branch admittances
        for branch in self.branches.values():
            if not branch.status:
                continue
                
            from_idx = bus_index[branch.from_bus]
            to_idx = bus_index[branch.to_bus]
            
            # Calculate branch admittance (DC approximation: only reactance)
            if branch.reactance == 0:
                warnings.warn(f"Branch {branch.id} has zero reactance, skipping")
                continue
                
            branch_admittance = -1j / branch.reactance  # Negative imaginary for DC
            
            # Fill admittance matrix
            Y[from_idx, to_idx] += branch_admittance
            Y[to_idx, from_idx] += branch_admittance
            Y[from_idx, from_idx] -= branch_admittance
            Y[to_idx, to_idx] -= branch_admittance
        
        # For DC analysis, we only need the imaginary part (negative of susceptance)
        self.admittance_matrix = -Y.imag
        return self.admittance_matrix
    
    def build_ptdf_matrix(self, slack_bus_id: int) -> np.ndarray:
        """
        Build Power Transfer Distribution Factor (PTDF) matrix.
        
        Args:
            slack_bus_id: ID of the slack bus
            
        Returns:
            PTDF matrix for line flow calculations
        """
        if self.admittance_matrix is None:
            self.build_admittance_matrix()
        
        n_buses = len(self.buses)
        bus_ids = sorted(self.buses.keys())
        bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        slack_idx = bus_index[slack_bus_id]
        
        # Remove slack bus row and column
        B_reduced = np.delete(self.admittance_matrix, slack_idx, axis=0)
        B_reduced = np.delete(B_reduced, slack_idx, axis=1)
        
        # Invert the reduced admittance matrix
        try:
            B_inv = np.linalg.inv(B_reduced)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            B_inv = np.linalg.pinv(B_reduced)
        
        # Build PTDF matrix
        active_branches = [b for b in self.branches.values() if b.status]
        n_branches = len(active_branches)
        ptdf = np.zeros((n_branches, n_buses - 1))  # Reduced dimensions
        
        branch_idx = 0
        for branch in active_branches:
            from_idx = bus_index[branch.from_bus]
            to_idx = bus_index[branch.to_bus]
            
            if branch.reactance != 0:
                susceptance = 1.0 / branch.reactance
                
                # Create angle difference vector for reduced system
                angle_diff_reduced = np.zeros(n_buses - 1)
                
                # Map bus indices to reduced system
                from_idx_reduced = from_idx if from_idx < slack_idx else from_idx - 1
                to_idx_reduced = to_idx if to_idx < slack_idx else to_idx - 1
                
                if from_idx != slack_idx:
                    angle_diff_reduced[from_idx_reduced] = 1.0
                if to_idx != slack_idx:
                    angle_diff_reduced[to_idx_reduced] = -1.0
                
                ptdf[branch_idx, :] = susceptance * (angle_diff_reduced @ B_inv)
            
            branch_idx += 1
        
        self.ptdf_matrix = ptdf
        return ptdf
    
    def solve_dc_opf(self, 
                     objective: str = 'cost',
                     carbon_price: float = 50.0,
                     power_balance_tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Solve DC Optimal Power Flow problem.
        
        Args:
            objective: Optimization objective ('cost', 'emissions', or 'combined')
            carbon_price: Price of carbon emissions ($/kg CO2)
            power_balance_tolerance: Tolerance for power balance constraints
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.generators:
            raise ValueError("No generators defined in the system")
        
        if not self.buses:
            raise ValueError("No buses defined in the system")
        
        # Find slack bus
        slack_buses = [bus for bus in self.buses.values() if bus.bus_type == BusType.SLACK]
        if not slack_buses:
            raise ValueError("No slack bus defined in the system")
        slack_bus = slack_buses[0]
        
        # Build system matrices
        self.build_admittance_matrix()
        self.build_ptdf_matrix(slack_bus.id)
        
        # Set up optimization variables
        n_generators = len(self.generators)
        generator_ids = list(self.generators.keys())
        
        # Generator power output variables
        pg = cp.Variable(n_generators, name="generator_power")
        
        # Bus voltage angles (slack bus angle is fixed at 0)
        n_buses = len(self.buses)
        bus_ids = sorted(self.buses.keys())
        bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
        slack_idx = bus_index[slack_bus.id]
        
        theta = cp.Variable(n_buses - 1, name="voltage_angles")
        
        # Constraints
        constraints = []
        
        # Generator limits
        for i, gen_id in enumerate(generator_ids):
            gen = self.generators[gen_id]
            if gen.status:
                constraints.append(pg[i] >= gen.real_power_min)
                constraints.append(pg[i] <= gen.real_power_max)
            else:
                constraints.append(pg[i] == 0)
        
        # Simplified power balance constraint: total generation = total demand
        total_demand = sum(bus.real_power_demand for bus in self.buses.values())
        constraints.append(cp.sum(pg) == total_demand)
        
        # Branch flow limits
        if self.ptdf_matrix is not None:
            branch_flows = self.ptdf_matrix @ theta  # theta is already reduced
            active_branches = [b for b in self.branches.values() if b.status]
            
            for i, branch in enumerate(active_branches):
                if branch.power_rating != float('inf'):
                    constraints.append(branch_flows[i] <= branch.power_rating)
                    constraints.append(branch_flows[i] >= -branch.power_rating)
        
        # Objective function
        if objective == 'cost':
            # Minimize generation cost
            cost_terms = []
            for i, gen_id in enumerate(generator_ids):
                gen = self.generators[gen_id]
                cost_terms.append(
                    gen.cost_coefficient_a * cp.square(pg[i]) +
                    gen.cost_coefficient_b * pg[i] +
                    gen.cost_coefficient_c
                )
            objective_function = cp.Minimize(cp.sum(cost_terms))
            
        elif objective == 'emissions':
            # Minimize carbon emissions
            emission_terms = []
            for i, gen_id in enumerate(generator_ids):
                gen = self.generators[gen_id]
                emission_terms.append(gen.carbon_emission_rate * pg[i])
            objective_function = cp.Minimize(cp.sum(emission_terms))
            
        elif objective == 'combined':
            # Minimize cost + carbon price * emissions
            combined_terms = []
            for i, gen_id in enumerate(generator_ids):
                gen = self.generators[gen_id]
                cost = (gen.cost_coefficient_a * cp.square(pg[i]) +
                       gen.cost_coefficient_b * pg[i] +
                       gen.cost_coefficient_c)
                emissions = gen.carbon_emission_rate * pg[i]
                combined_terms.append(cost + carbon_price * emissions)
            objective_function = cp.Minimize(cp.sum(combined_terms))
            
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Solve optimization problem
        problem = cp.Problem(objective_function, constraints)
        
        try:
            problem.solve(solver=cp.ECOS)
        except Exception:
            try:
                problem.solve(solver=cp.SCS)
            except Exception:
                problem.solve()
        
        # Extract results
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        # Store results
        self.generator_dispatch = {}
        total_cost = 0.0
        total_emissions = 0.0
        
        for i, gen_id in enumerate(generator_ids):
            power_output = pg[i].value if pg[i].value is not None else 0.0
            self.generator_dispatch[gen_id] = power_output
            
            # Calculate cost and emissions
            gen = self.generators[gen_id]
            gen_cost = (gen.cost_coefficient_a * power_output**2 +
                       gen.cost_coefficient_b * power_output +
                       gen.cost_coefficient_c)
            gen_emissions = gen.carbon_emission_rate * power_output
            
            total_cost += gen_cost
            total_emissions += gen_emissions
        
        self.total_cost = total_cost
        self.total_emissions = total_emissions
        
        # Voltage angles (add slack bus angle = 0)
        if theta.value is not None:
            full_angles = np.zeros(n_buses)
            angle_idx = 0
            for i in range(n_buses):
                if i != slack_idx:
                    full_angles[i] = theta.value[angle_idx]
                    angle_idx += 1
            self.voltage_angles = full_angles
        
        # Calculate power flows
        if self.ptdf_matrix is not None and theta.value is not None:
            self.power_flows = self.ptdf_matrix @ theta.value
        
        return {
            'status': problem.status,
            'objective_value': problem.value,
            'generator_dispatch': self.generator_dispatch,
            'total_cost': self.total_cost,
            'total_emissions': self.total_emissions,
            'voltage_angles': self.voltage_angles,
            'power_flows': self.power_flows
        }
    
    def calculate_carbon_footprint(self) -> Dict[str, float]:
        """
        Calculate detailed carbon footprint metrics.
        
        Returns:
            Dictionary with carbon footprint analysis
        """
        if self.generator_dispatch is None:
            raise ValueError("No solution available. Run solve_dc_opf() first.")
        
        carbon_metrics = {
            'total_emissions_kg_co2': self.total_emissions,
            'emissions_per_generator': {},
            'emission_rate_weighted_avg': 0.0,
            'clean_energy_percentage': 0.0
        }
        
        total_generation = 0.0
        clean_generation = 0.0
        weighted_emission_rate = 0.0
        
        for gen_id, power_output in self.generator_dispatch.items():
            gen = self.generators[gen_id]
            gen_emissions = gen.carbon_emission_rate * power_output
            
            carbon_metrics['emissions_per_generator'][gen_id] = {
                'power_mw': power_output,
                'emissions_kg_co2': gen_emissions,
                'emission_rate_kg_co2_per_mwh': gen.carbon_emission_rate
            }
            
            total_generation += power_output
            weighted_emission_rate += gen.carbon_emission_rate * power_output
            
            # Consider generators with emission rate < 0.1 kg CO2/MWh as "clean"
            if gen.carbon_emission_rate < 0.1:
                clean_generation += power_output
        
        if total_generation > 0:
            carbon_metrics['emission_rate_weighted_avg'] = weighted_emission_rate / total_generation
            carbon_metrics['clean_energy_percentage'] = max(0.0, (clean_generation / total_generation) * 100)
        
        return carbon_metrics
    
    def export_results_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Export optimization results to pandas DataFrames.
        
        Returns:
            Dictionary containing DataFrames with results
        """
        if self.generator_dispatch is None:
            raise ValueError("No solution available. Run solve_dc_opf() first.")
        
        # Generator results
        gen_data = []
        for gen_id, power_output in self.generator_dispatch.items():
            gen = self.generators[gen_id]
            gen_cost = (gen.cost_coefficient_a * power_output**2 +
                       gen.cost_coefficient_b * power_output +
                       gen.cost_coefficient_c)
            gen_emissions = gen.carbon_emission_rate * power_output
            
            gen_data.append({
                'generator_id': gen_id,
                'bus_id': gen.bus_id,
                'power_output_mw': power_output,
                'min_power_mw': gen.real_power_min,
                'max_power_mw': gen.real_power_max,
                'cost_usd': gen_cost,
                'emissions_kg_co2': gen_emissions,
                'emission_rate_kg_co2_per_mwh': gen.carbon_emission_rate
            })
        
        gen_df = pd.DataFrame(gen_data)
        
        # Bus results
        bus_data = []
        bus_ids = sorted(self.buses.keys())
        
        for i, bus_id in enumerate(bus_ids):
            bus = self.buses[bus_id]
            voltage_angle = self.voltage_angles[i] if self.voltage_angles is not None else 0.0
            
            bus_data.append({
                'bus_id': bus_id,
                'bus_name': bus.name,
                'bus_type': bus.bus_type.value,
                'voltage_angle_rad': voltage_angle,
                'voltage_angle_deg': np.degrees(voltage_angle),
                'real_power_demand_mw': bus.real_power_demand,
                'reactive_power_demand_mvar': bus.reactive_power_demand
            })
        
        bus_df = pd.DataFrame(bus_data)
        
        # Branch flow results
        branch_data = []
        active_branches = [b for b in self.branches.values() if b.status]
        
        for i, branch in enumerate(active_branches):
            power_flow = self.power_flows[i] if self.power_flows is not None else 0.0
            
            branch_data.append({
                'branch_id': branch.id,
                'from_bus': branch.from_bus,
                'to_bus': branch.to_bus,
                'power_flow_mw': power_flow,
                'power_rating_mva': branch.power_rating,
                'loading_percentage': abs(power_flow) / branch.power_rating * 100 
                                    if branch.power_rating != float('inf') else 0.0
            })
        
        branch_df = pd.DataFrame(branch_data)
        
        return {
            'generators': gen_df,
            'buses': bus_df,
            'branches': branch_df
        }


def create_sample_system() -> DCOptimalPowerFlow:
    """
    Create a sample 3-bus power system for testing.
    
    Returns:
        Configured DCOptimalPowerFlow instance
    """
    dc_opf = DCOptimalPowerFlow(base_mva=100.0)
    
    # Add buses
    dc_opf.add_bus(Bus(
        id=1, name="Bus_1", bus_type=BusType.SLACK,
        voltage_magnitude=1.0, voltage_angle=0.0,
        real_power_demand=0.0, reactive_power_demand=0.0
    ))
    
    dc_opf.add_bus(Bus(
        id=2, name="Bus_2", bus_type=BusType.PQ,
        voltage_magnitude=1.0, voltage_angle=0.0,
        real_power_demand=100.0, reactive_power_demand=50.0
    ))
    
    dc_opf.add_bus(Bus(
        id=3, name="Bus_3", bus_type=BusType.PQ,
        voltage_magnitude=1.0, voltage_angle=0.0,
        real_power_demand=80.0, reactive_power_demand=40.0
    ))
    
    # Add branches
    dc_opf.add_branch(Branch(
        id=1, from_bus=1, to_bus=2,
        resistance=0.01, reactance=0.1, susceptance=0.02,
        power_rating=200.0
    ))
    
    dc_opf.add_branch(Branch(
        id=2, from_bus=1, to_bus=3,
        resistance=0.02, reactance=0.15, susceptance=0.015,
        power_rating=150.0
    ))
    
    dc_opf.add_branch(Branch(
        id=3, from_bus=2, to_bus=3,
        resistance=0.015, reactance=0.12, susceptance=0.01,
        power_rating=100.0
    ))
    
    # Add generators
    dc_opf.add_generator(Generator(
        id=1, bus_id=1,
        real_power_min=0.0, real_power_max=200.0,
        reactive_power_min=-50.0, reactive_power_max=100.0,
        cost_coefficient_a=0.01, cost_coefficient_b=20.0, cost_coefficient_c=100.0,
        carbon_emission_rate=0.8  # High carbon coal plant
    ))
    
    dc_opf.add_generator(Generator(
        id=2, bus_id=2,
        real_power_min=0.0, real_power_max=150.0,
        reactive_power_min=-30.0, reactive_power_max=80.0,
        cost_coefficient_a=0.02, cost_coefficient_b=30.0, cost_coefficient_c=50.0,
        carbon_emission_rate=0.4  # Natural gas plant
    ))
    
    dc_opf.add_generator(Generator(
        id=3, bus_id=3,
        real_power_min=0.0, real_power_max=100.0,
        reactive_power_min=-20.0, reactive_power_max=60.0,
        cost_coefficient_a=0.0, cost_coefficient_b=50.0, cost_coefficient_c=0.0,
        carbon_emission_rate=0.0  # Clean renewable energy
    ))
    
    return dc_opf


if __name__ == "__main__":
    # Example usage and testing
    print("Testing DC Optimal Power Flow with Python 3.13 compatibility...")
    
    # Create sample system
    dc_opf = create_sample_system()
    
    print(f"Created system with {len(dc_opf.buses)} buses, "
          f"{len(dc_opf.branches)} branches, and {len(dc_opf.generators)} generators")
    
    # Solve for minimum cost
    print("\nSolving for minimum cost...")
    results_cost = dc_opf.solve_dc_opf(objective='cost')
    print(f"Status: {results_cost['status']}")
    print(f"Total cost: ${results_cost['total_cost']:.2f}")
    print(f"Total emissions: {results_cost['total_emissions']:.2f} kg CO2")
    
    # Solve for minimum emissions
    print("\nSolving for minimum emissions...")
    results_emissions = dc_opf.solve_dc_opf(objective='emissions')
    print(f"Status: {results_emissions['status']}")
    print(f"Total cost: ${results_emissions['total_cost']:.2f}")
    print(f"Total emissions: {results_emissions['total_emissions']:.2f} kg CO2")
    
    # Solve for combined objective
    print("\nSolving for combined cost and emissions (carbon price = $50/kg CO2)...")
    results_combined = dc_opf.solve_dc_opf(objective='combined', carbon_price=50.0)
    print(f"Status: {results_combined['status']}")
    print(f"Total cost: ${results_combined['total_cost']:.2f}")
    print(f"Total emissions: {results_combined['total_emissions']:.2f} kg CO2")
    
    # Calculate carbon footprint
    carbon_metrics = dc_opf.calculate_carbon_footprint()
    print(f"\nCarbon footprint analysis:")
    print(f"Total emissions: {carbon_metrics['total_emissions_kg_co2']:.2f} kg CO2")
    print(f"Weighted average emission rate: {carbon_metrics['emission_rate_weighted_avg']:.3f} kg CO2/MWh")
    print(f"Clean energy percentage: {carbon_metrics['clean_energy_percentage']:.1f}%")
    
    # Export results
    dataframes = dc_opf.export_results_to_dataframe()
    print(f"\nGenerator dispatch:")
    print(dataframes['generators'][['generator_id', 'power_output_mw', 'cost_usd', 'emissions_kg_co2']])
    
    print("\nDC-OPF testing completed successfully!")