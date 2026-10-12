"""Unit tests for the Evolutionary Algorithm (AE) optimizer."""

import numpy as np
import pytest

from src.functions.base import ObjectiveFunction
from src.functions.rosenbrock import Rosenbrock
from src.functions.rastrigin import Rastrigin
from src.metrics.evaluation_counter import CountedFunction
from src.optimizers.AE import (
    CONFIG_POR_DEFECTO,
    ajustar_limites,
    crossover_blx_alpha,
    elitismo,
    inicializar_poblacion,
    mutacion_gaussiana,
    optimize,
    preparar_limites,
    seleccion_torneo,
    validar_configuracion,
)


# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------

class Sphere(ObjectiveFunction):
    """Convex quadratic: f(x) = sum(x_i**2), with minimum f* = 0."""

    def __init__(self):
        super().__init__(
            name="Sphere",
            default_bounds=(-5.0, 5.0),
            global_minimum_f=0.0,
        )

    def evaluate(self, x):
        return float(np.sum(self._validate_input(x) ** 2))

    def gradient(self, x):
        return 2.0 * self._validate_input(x)

    def global_minimum_x(self, dimension):
        return np.zeros(dimension)


class ShiftedSphere(Sphere):
    """Sphere shifted upward by 5, so its global minimum value is 5."""

    def __init__(self):
        ObjectiveFunction.__init__(
            self,
            name="ShiftedSphere",
            default_bounds=(-5.0, 5.0),
            global_minimum_f=5.0,
        )

    def evaluate(self, x):
        return float(np.sum(self._validate_input(x) ** 2)) + 5.0


class Spy(ObjectiveFunction):
    """Records all objective-function evaluations."""

    def __init__(self, base):
        super().__init__(
            name=base.name,
            default_bounds=base.default_bounds,
            global_minimum_f=base.global_minimum_f,
        )
        self.base = base
        self.f_points = []
        self.f_values = []
        self.grad_points = []

    def evaluate(self, x):
        point = np.array(x, dtype=float).copy()
        value = float(self.base.evaluate(point))
        self.f_points.append(point)
        self.f_values.append(value)
        return value

    def gradient(self, x):
        point = np.array(x, dtype=float).copy()
        self.grad_points.append(point)
        return self.base.gradient(point)

    def global_minimum_x(self, dimension):
        return self.base.global_minimum_x(dimension)


@pytest.fixture
def config():
    return {
        "population_size": 20,
        "tournament_size": 3,
        "crossover_rate": 0.9,
        "alpha": 0.5,
        "mutation_rate": 0.1,
        "mutation_sigma_rel": 0.1,
        "n_elites": 2,
        "tolerance": 1e-4,
    }


# ---------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------

class TestAEConfiguration:

    def test_default_configuration_is_valid(self):
        result = validar_configuracion(None)

        assert result == CONFIG_POR_DEFECTO

    def test_partial_configuration_preserves_defaults(self):
        result = validar_configuracion({"population_size": 12})

        assert result["population_size"] == 12
        assert result["mutation_rate"] == CONFIG_POR_DEFECTO["mutation_rate"]
        assert result["tournament_size"] == CONFIG_POR_DEFECTO["tournament_size"]

    @pytest.mark.parametrize("population_size", [0, 1, -1, 2.5, True])
    def test_invalid_population_size(self, population_size):
        with pytest.raises(ValueError):
            validar_configuracion({"population_size": population_size})

    @pytest.mark.parametrize("tournament_size", [0, -1, 21, 1.5, True])
    def test_invalid_tournament_size(self, tournament_size):
        with pytest.raises(ValueError):
            validar_configuracion({
                "population_size": 20,
                "tournament_size": tournament_size,
            })

    @pytest.mark.parametrize("n_elites", [-1, 20, 1.5, True])
    def test_invalid_elite_count(self, n_elites):
        with pytest.raises(ValueError):
            validar_configuracion({
                "population_size": 20,
                "n_elites": n_elites,
            })

    @pytest.mark.parametrize(
        "key,value",
        [
            ("crossover_rate", -0.1),
            ("crossover_rate", 1.1),
            ("crossover_rate", float("nan")),
            ("mutation_rate", -0.1),
            ("mutation_rate", 1.1),
            ("mutation_rate", float("nan")),
            ("alpha", -0.1),
            ("alpha", float("nan")),
            ("mutation_sigma_rel", 0.0),
            ("mutation_sigma_rel", -0.1),
            ("mutation_sigma_rel", float("nan")),
            ("tolerance", -0.1),
            ("tolerance", float("nan")),
        ],
    )
    def test_invalid_numeric_configuration(self, key, value):
        with pytest.raises((ValueError, TypeError)):
            validar_configuracion({key: value})

    @pytest.mark.parametrize("max_generations", [-1, 1.5, True])
    def test_invalid_max_generations(self, max_generations):
        with pytest.raises(ValueError):
            validar_configuracion({"max_generations": max_generations})

    def test_zero_max_generations_is_valid(self):
        result = validar_configuracion({"max_generations": 0})

        assert result["max_generations"] == 0

    def test_unknown_configuration_key_is_rejected(self):
        with pytest.raises(ValueError):
            validar_configuracion({"unknown_parameter": 1})

    def test_non_dictionary_configuration_is_rejected(self):
        with pytest.raises(ValueError):
            validar_configuracion(["population_size", 20])


# ---------------------------------------------------------
# Bounds and input validation
# ---------------------------------------------------------

class TestAEValidation:

    def test_scalar_bounds_are_expanded_to_dimension(self):
        lower, upper = preparar_limites((-5.0, 10.0), 3)

        np.testing.assert_array_equal(lower, [-5.0, -5.0, -5.0])
        np.testing.assert_array_equal(upper, [10.0, 10.0, 10.0])

    def test_per_dimension_bounds_are_preserved(self):
        bounds = [(0.0, 1.0), (-10.0, -5.0)]
        lower, upper = preparar_limites(bounds, 2)

        np.testing.assert_array_equal(lower, [0.0, -10.0])
        np.testing.assert_array_equal(upper, [1.0, -5.0])

    @pytest.mark.parametrize(
        "bounds",
        [
            (1.0, 1.0),
            (5.0, -5.0),
            (float("-inf"), 5.0),
            (-5.0, float("inf")),
            [(0.0, 1.0), (2.0, 2.0)],
            np.zeros((3, 3)),
        ],
    )
    def test_invalid_bounds_are_rejected(self, bounds):
        with pytest.raises(ValueError):
            preparar_limites(bounds, 2)

    def test_incompatible_function_type_is_rejected(self):
        with pytest.raises(TypeError):
            optimize(
                lambda x: float(np.sum(x**2)),
                (-5.0, 5.0),
                2,
                seed=0,
                budget=20,
            )

    @pytest.mark.parametrize(
        "dimension,seed,budget",
        [
            (0, 0, 20),
            (-1, 0, 20),
            (2.5, 0, 20),
            (True, 0, 20),
            (2, 1.5, 20),
            (2, True, 20),
            (2, 0, 0),
            (2, 0, -1),
            (2, 0, 1.5),
            (2, 0, True),
        ],
    )
    def test_invalid_optimize_arguments(self, dimension, seed, budget):
        with pytest.raises(ValueError):
            optimize(
                Rosenbrock(),
                (-5.0, 10.0),
                dimension,
                seed=seed,
                budget=budget,
            )

    def test_counted_function_dimension_must_match(self):
        counter = CountedFunction(Rosenbrock(), 2)

        with pytest.raises(ValueError):
            optimize(
                counter,
                (-5.0, 10.0),
                3,
                seed=0,
                budget=100,
            )


# ---------------------------------------------------------
# Operator tests
# ---------------------------------------------------------

class TestAEOperators:

    def test_initial_population_has_correct_shape_and_bounds(self):
        rng = np.random.default_rng(42)
        lower = np.array([-5.0, 0.0, -2.0])
        upper = np.array([5.0, 1.0, 2.0])

        population = inicializar_poblacion(30, lower, upper, rng)

        assert population.shape == (30, 3)
        assert np.all(population >= lower)
        assert np.all(population <= upper)

    def test_initial_population_is_reproducible(self):
        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        pop1 = inicializar_poblacion(
            10, lower, upper, np.random.default_rng(42)
        )
        pop2 = inicializar_poblacion(
            10, lower, upper, np.random.default_rng(42)
        )

        np.testing.assert_array_equal(pop1, pop2)

    def test_tournament_selection_returns_valid_indices(self):
        fitness = np.array([5.0, 1.0, 3.0, 4.0])
        rng = np.random.default_rng(42)

        parents = seleccion_torneo(fitness, 10, 2, rng)

        assert parents.shape == (10,)
        assert np.all(parents >= 0)
        assert np.all(parents < len(fitness))

    def test_tournament_size_one_selects_best_of_each_singleton(self):
        fitness = np.array([5.0, 1.0, 3.0])
        rng = np.random.default_rng(42)

        parents = seleccion_torneo(fitness, 20, 1, rng)

        assert np.all(np.isin(parents, [0, 1, 2]))

    def test_crossover_disabled_returns_parent_copies(self):
        parent_a = np.array([1.0, 2.0])
        parent_b = np.array([3.0, 4.0])

        child_a, child_b = crossover_blx_alpha(
            parent_a,
            parent_b,
            alpha=0.5,
            tasa_crossover=0.0,
            generador=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(child_a, parent_a)
        np.testing.assert_array_equal(child_b, parent_b)

        assert child_a is not parent_a
        assert child_b is not parent_b

    def test_blx_alpha_children_stay_inside_expanded_interval(self):
        parent_a = np.array([1.0, 4.0])
        parent_b = np.array([3.0, 2.0])
        alpha = 0.5

        child_a, child_b = crossover_blx_alpha(
            parent_a,
            parent_b,
            alpha=alpha,
            tasa_crossover=1.0,
            generador=np.random.default_rng(42),
        )

        low = np.minimum(parent_a, parent_b) - alpha * np.abs(parent_a - parent_b)
        high = np.maximum(parent_a, parent_b) + alpha * np.abs(parent_a - parent_b)

        assert np.all(child_a >= low)
        assert np.all(child_a <= high)
        assert np.all(child_b >= low)
        assert np.all(child_b <= high)

    def test_mutation_rate_zero_does_not_change_children(self):
        children = np.array([[1.0, 2.0], [3.0, 4.0]])

        mutated = mutacion_gaussiana(
            children,
            tasa_mutacion=0.0,
            sigma=np.array([1.0, 1.0]),
            generador=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(mutated, children)

    def test_mutation_rate_one_mutates_all_genes_for_nonzero_noise(self):
        children = np.zeros((5, 3))

        mutated = mutacion_gaussiana(
            children,
            tasa_mutacion=1.0,
            sigma=np.ones(3),
            generador=np.random.default_rng(42),
        )

        assert mutated.shape == children.shape
        assert np.all(np.isfinite(mutated))
        assert np.all(mutated != 0.0)

    def test_adjust_bounds_clips_each_dimension(self):
        population = np.array([
            [-10.0, 0.5],
            [0.5, 20.0],
            [0.5, 0.5],
        ])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        adjusted = ajustar_limites(population, lower, upper)

        assert np.all(adjusted >= lower)
        assert np.all(adjusted <= upper)
        np.testing.assert_array_equal(
            adjusted,
            [[0.0, 0.5], [0.5, 1.0], [0.5, 0.5]],
        )

    def test_elitism_keeps_best_individuals_in_order(self):
        population = np.array([[0.0], [1.0], [2.0], [3.0]])
        fitness = np.array([4.0, 1.0, 3.0, 2.0])

        elite_pop, elite_fitness = elitismo(population, fitness, 2)

        np.testing.assert_array_equal(elite_pop, [[1.0], [3.0]])
        np.testing.assert_array_equal(elite_fitness, [1.0, 2.0])

    def test_elitism_with_zero_elites_returns_empty_arrays(self):
        population = np.array([[0.0], [1.0]])
        fitness = np.array([0.0, 1.0])

        elite_pop, elite_fitness = elitismo(population, fitness, 0)

        assert elite_pop.shape == (0, 1)
        assert elite_fitness.shape == (0,)


# ---------------------------------------------------------
# Optimizer result contract and budget
# ---------------------------------------------------------

class TestAEOptimizer:

    @pytest.mark.parametrize(
        "FunctionClass,dim,bounds",
        [
            (Rosenbrock, 2, (-5.0, 10.0)),
            (Rosenbrock, 3, (-5.0, 10.0)),
            (Rastrigin, 2, (-5.12, 5.12)),
            (Rastrigin, 3, (-5.12, 5.12)),
        ],
    )
    def test_common_interface_schema(self, FunctionClass, dim, bounds):
        result = optimize(
            FunctionClass(bounds=bounds),
            bounds,
            dim,
            seed=42,
            budget=500,
            config={"population_size": 20},
        )

        required_keys = [
            "best_x",
            "best_f",
            "f_evaluations",
            "gradient_evaluations",
            "equivalent_evaluations",
            "iterations",
            "success",
            "history",
        ]

        for key in required_keys:
            assert key in result

        assert isinstance(result["best_x"], list)
        assert len(result["best_x"]) == dim
        assert isinstance(result["best_f"], float)
        assert np.isfinite(result["best_f"])
        assert isinstance(result["f_evaluations"], int)
        assert isinstance(result["gradient_evaluations"], int)
        assert isinstance(result["equivalent_evaluations"], int)
        assert isinstance(result["iterations"], int)
        assert isinstance(result["success"], bool)
        assert isinstance(result["history"], list)
        assert result["history"]

        history = result["history"]

        assert [entry["step"] for entry in history] == list(
            range(result["iterations"] + 1)
        )
        assert len(history) == result["iterations"] + 1

        history_f = [entry["best_f"] for entry in history]
        assert all(b <= a for a, b in zip(history_f, history_f[1:]))

        assert history[-1]["best_f"] == result["best_f"]
        assert history[-1]["equivalent_evaluations"] == result["equivalent_evaluations"]

        for entry in history:
            assert len(entry["best_x"]) == dim
            assert "f_evaluations" in entry

    @pytest.mark.parametrize(
        "budget",
        [1, 2, 5, 10, 19, 20, 21, 25, 49, 50, 51, 100, 499, 500],
    )
    def test_strict_budget_compliance(self, budget):
        result = optimize(
            Rosenbrock(),
            (-5.0, 10.0),
            2,
            seed=10,
            budget=budget,
            config={"population_size": 20},
        )

        assert result["equivalent_evaluations"] <= budget
        assert result["f_evaluations"] <= budget
        assert result["gradient_evaluations"] == 0
        assert result["equivalent_evaluations"] == result["f_evaluations"]

    def test_initial_population_is_reduced_to_fit_budget(self):
        spy = Spy(Rastrigin())

        result = optimize(
            spy,
            (-5.12, 5.12),
            2,
            seed=0,
            budget=7,
            config={"population_size": 50},
        )

        assert result["f_evaluations"] == 7
        assert result["iterations"] == 0
        assert len(spy.f_points) == 7
        assert np.isfinite(result["best_f"])

    def test_every_evaluated_point_stays_within_bounds(self):
        lower = np.array([-5.0, -2.0, 0.0])
        upper = np.array([10.0, 2.0, 1.0])
        bounds = list(zip(lower, upper))
        spy = Spy(Rosenbrock())

        result = optimize(
            spy,
            bounds,
            3,
            seed=42,
            budget=500,
            config={"population_size": 20},
        )

        points = np.asarray(spy.f_points)

        assert len(points) == result["f_evaluations"]
        assert np.all(points >= lower)
        assert np.all(points <= upper)
        assert np.all(np.asarray(result["best_x"]) >= lower)
        assert np.all(np.asarray(result["best_x"]) <= upper)

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_reproducibility(self, seed):
        config = {"population_size": 15, "record_population": True}

        result_1 = optimize(
            Rosenbrock(), (-5.0, 10.0), 2, seed, 300, config
        )
        result_2 = optimize(
            Rosenbrock(), (-5.0, 10.0), 2, seed, 300, config
        )

        assert result_1["best_x"] == result_2["best_x"]
        assert result_1["best_f"] == result_2["best_f"]
        assert result_1["equivalent_evaluations"] == result_2["equivalent_evaluations"]
        assert result_1["iterations"] == result_2["iterations"]
        assert result_1["history"] == result_2["history"]

    def test_record_population_option(self):
        population_size = 10
        budget = 40
        common = {
            "population_size": population_size,
            "record_population": True,
        }

        with_population = optimize(
            Rastrigin(),
            (-5.12, 5.12),
            2,
            seed=42,
            budget=budget,
            config=common,
        )
        without_population = optimize(
            Rastrigin(),
            (-5.12, 5.12),
            2,
            seed=42,
            budget=budget,
            config={
                "population_size": population_size,
                "record_population": False,
            },
        )

        for entry in with_population["history"]:
            assert "population" in entry
            recorded = np.asarray(entry["population"])
            assert recorded.ndim == 2
            assert recorded.shape[1] == 2
            assert len(recorded) >= 1
            assert len(recorded) <= population_size

        for entry in without_population["history"]:
            assert "population" not in entry

        assert with_population["best_x"] == without_population["best_x"]
        assert with_population["best_f"] == without_population["best_f"]

    def test_recording_population_does_not_change_optimization(self):
        config = {"population_size": 10}

        result_a = optimize(
            Rastrigin(), (-5.12, 5.12), 2, 42, 100,
            config={**config, "record_population": False},
        )
        result_b = optimize(
            Rastrigin(), (-5.12, 5.12), 2, 42, 100,
            config={**config, "record_population": True},
        )

        assert result_a["best_x"] == result_b["best_x"]
        assert result_a["best_f"] == result_b["best_f"]
        assert result_a["history"][-1]["equivalent_evaluations"] == (
            result_b["history"][-1]["equivalent_evaluations"]
        )

    def test_max_generations_zero_stops_after_initial_population(self):
        result = optimize(
            Rastrigin(),
            (-5.12, 5.12),
            2,
            seed=42,
            budget=100,
            config={
                "population_size": 10,
                "max_generations": 0,
            },
        )

        assert result["iterations"] == 0
        assert result["f_evaluations"] == 10
        assert len(result["history"]) == 1

    def test_max_generations_limits_iterations(self):
        result = optimize(
            Rastrigin(),
            (-5.12, 5.12),
            2,
            seed=42,
            budget=500,
            config={
                "population_size": 10,
                "max_generations": 3,
            },
        )

        assert result["iterations"] == 3
        assert len(result["history"]) == 4

    def test_reused_counter_respects_remaining_budget(self):
        counter = CountedFunction(Rosenbrock(), 2)

        for _ in range(5):
            counter.evaluate(np.array([0.0, 0.0]))

        result = optimize(
            counter,
            (-5.0, 10.0),
            2,
            seed=0,
            budget=100,
            config={"population_size": 10},
        )

        assert counter.equivalent_evaluations <= 100
        assert result["equivalent_evaluations"] == counter.equivalent_evaluations
        assert result["f_evaluations"] == counter.f_evaluations

    def test_already_exhausted_budget_is_rejected(self):
        counter = CountedFunction(Rosenbrock(), 2)

        for _ in range(10):
            counter.evaluate(np.array([0.0, 0.0]))

        with pytest.raises(ValueError):
            optimize(
                counter,
                (-5.0, 10.0),
                2,
                seed=0,
                budget=10,
                config={"population_size": 10},
            )

        assert counter.equivalent_evaluations == 10

    def test_success_uses_function_global_minimum_value(self):
        result = optimize(
            ShiftedSphere(),
            (-5.0, 5.0),
            2,
            seed=0,
            budget=10,
            config={
                "population_size": 10,
                "tolerance": 1e6,
            },
        )

        assert result["success"] is True

    def test_success_is_false_when_tolerance_is_too_strict(self):
        result = optimize(
            Rastrigin(),
            (-5.12, 5.12),
            2,
            seed=0,
            budget=10,
            config={
                "population_size": 10,
                "tolerance": 0.0,
            },
        )

        assert result["success"] is False

    def test_best_solution_matches_best_value_seen(self):
        spy = Spy(Rastrigin())

        result = optimize(
            spy,
            (-5.12, 5.12),
            2,
            seed=8,
            budget=200,
            config={"population_size": 20},
        )

        assert result["best_f"] == min(spy.f_values)

        best_x = np.asarray(result["best_x"])
        best_was_evaluated = any(
            np.array_equal(best_x, point) for point in spy.f_points
        )
        assert best_was_evaluated

    def test_population_recording_contains_points_within_bounds(self):
        result = optimize(
            Rastrigin(),
            (-5.12, 5.12),
            2,
            seed=42,
            budget=100,
            config={
                "population_size": 10,
                "record_population": True,
            },
        )

        for entry in result["history"]:
            population = np.asarray(entry["population"])
            assert np.all(population >= -5.12)
            assert np.all(population <= 5.12)

    def test_non_finite_initial_objective_values_raise(self):
        class NonFiniteFunction(ObjectiveFunction):
            def __init__(self):
                super().__init__(
                    name="NonFinite",
                    default_bounds=(-1.0, 1.0),
                    global_minimum_f=0.0,
                )

            def evaluate(self, x):
                return float("nan")

            def gradient(self, x):
                return np.zeros_like(x)

            def global_minimum_x(self, dimension):
                return np.zeros(dimension)

        with pytest.raises(ValueError):
            optimize(
                NonFiniteFunction(),
                (-1.0, 1.0),
                2,
                seed=0,
                budget=10,
                config={"population_size": 5},
            )

    def test_converges_on_simple_convex_quadratic(self):
        result = optimize(
            Sphere(),
            (-5.0, 5.0),
            2,
            seed=0,
            budget=5000,
            config={
                "population_size": 30,
                "tolerance": 1e-4,
            },
        )

        assert result["best_f"] >= 0.0
        assert result["best_f"] < 1e-2
        assert np.all(np.asarray(result["best_x"]) >= -5.0)
        assert np.all(np.asarray(result["best_x"]) <= 5.0)
