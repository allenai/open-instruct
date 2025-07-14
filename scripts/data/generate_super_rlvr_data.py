"""
Used to generate a super-rlvr dataset for use with Zhiyuan's verifiers.

export tasks=(
  "ABProgramSimulation"
  "AddMultiple_Divisible_Counting"
  "AdditionTable"
  "AndOr_Sequence_Counting"
  "Axis_KCenter"
  "BezoutIdentity"
  "Binario"
  "Binario_NoAdjacencyRequirement"
  "BinaryAlternation"
  "BitEquationCounting"
  "BitAndZero_PathCounting"
  "BlockImage"
  "BoundedIntervalIntersection"
  "BoundedMeanSubarrayCounting"
  "BoundedSubarrayCounting"
  "Bridge"
  "BucketSorting"
  "CampfireParty"
  "CampsitePuzzle"
  "CardColoringCounting"
  "Cinema"
  "Circuit"
  "CirculatingDecimalCounting"
  "ColoringCounting"
  "CombinationOddSubsequenceCounting"
  "CongruentEquation"
  "ConstructHackInterval"
  "ConvexHull"
  "CountdownEqual"
  "CountdownClose"
  "CRT"
  "Cryptarithmetic"
  "CycleCounting"
  "DecreasingDigitCounting"
  "DegreeFixed_SpanningTree"
  "DeltaMinPopcount"
  "DifferenceConstraintSystem"
  "Differentiate"
  "DigitLISCounting"
  "DiscreteLogarithm"
  "Division"
  "DivisorFlipExpectation"
  "DoublePalindromicStringCounting"
  "DoubleStackSorting"
  "EightDigitPuzzle"
  "EuclidGame"
  "Expression_AddingParenthese_Counting"
  "FBI_BinaryTree"
  "Fibonacci"
  "Fibtrain"
  "FixedModK_Selection_Counting"
  "FractionalProgramming"
  "FutoshikiPuzzle"
  "GaussianElimination"
  "GcdLcmCounting"
  "GCDOne_Counting"
  "GCDPrime_Counting"
  "GraphContainTreeCounting"
  "GraphIsomorphism"
  "GridBFS"
  "GridComponent"
  "HalvingChainCounting"
  "HamiltonianPath"
  "HamiltonianPathExistence"
  "HeapCounting"
  "HitoriPuzzle"
  "IntegerFactorizationCounting"
  "IntegerProgramming"
  "Integral"
  "InversionPairK_Counting"
  "Josephus"
  "JugPuzzle"
  "KPartition"
  "Kakurasu"
  "KingSorting"
  "Knapsack"
  "KnightsAndKnaves"
  "Kth_BinaryTree"
  "Kth_SemiBalancedBracketSequence"
  "LCM"
  "LDSTwo_Counting"
  "LightUpPuzzle"
  "LIS_LDS_Concatenation"
  "Longest_DoublePalindrome"
  "Longest_MatchingSubsequence"
  "LongestPath"
  "MagicSquarePuzzle"
  "Matrix_BinaryExponentiation"
  "MatrixPermutation_BothDiagonalOne"
  "MatrixPermutationEquivalence"
  "MatrixPermutation_MainDiagonalOne"
  "MatrixPooling"
  "MaxMultSplit"
  "MaxPermutation"
  "MaxThreeSquareSum"
  "MaxTreeXorPath"
  "MaxXorPath"
  "MaxXorSet"
  "MaximumAchromaticNumber"
  "MaximumClique"
  "MaximumDivisor"
  "Maximum_IndependentSet_Tree"
  "Maximum_SubsequenceNum"
  "MaximumWeightMatching"
  "Maze"
  "MinKDivisorNumber"
  "MinNoSolutionLinearDiophantineEquation"
  "MinNonsubstring"
  "MinXorPair"
  "Minesweeping"
  "MinimumChromaticNumber"
  "MinimumCost_MaximumFlow"
  "Minimum_CrossingEdges_GraphPartition"
  "MinimumDirectedSpanningTree"
  "Minimum_DominatingInterval"
  "Minimum_DominatingSet"
  "MinimumHarmoniousChromaticNumber"
  "Minimum_MaxSlicer"
  "MinimumRatioPath"
  "MinimumSpanningTree"
  "MinimumSpanningTreeCounting"
  "MinimumSteinerTree"
  "MinimumSumDifferenceSubmatrix"
  "MinimumTreeWeightedDominatingAncestor"
  "Minimum_VertexCover"
  "MinimumWeightedSpanningTree"
  "MonochromeBlockCounting"
  "MultipleFlippingGame"
  "Multiplication"
  "NANDResultCounting"
  "NegativeBase"
  "NextPalindromic"
  "NinePuzzle"
  "NoDoubleTripleCounting"
  "NumberPartitionCounting"
  "Numbrix"
  "OddVisitation"
  "PairMoreOneCounting"
  "PalindromePartitionCounting"
  "Path_NoGoingBack_Counting"
  "PCPPermutation"
  "PipelineArrangement"
  "PolynomialFactorization"
  "PolynomialInterpolation"
  "PolynomialMinimum"
  "PolynomialRemainder"
  "PowerCycle"
  "PowerShortcut"
  "PowerNest"
  "PrefixConcatenation"
  "PreorderTraversal"
  "PrimeGraph_MinimumChromaticNumber"
  "QuadMagicItems"
  "QuantumLockPuzzle"
  "QueenPlacement"
  "RandomRangeMaxExpectation"
  "RecursiveFunction"
  "RootExtraction"
  "RoyalLockCounting"
  "SalesmanFatigue"
  "SameAdjacencyCounting"
  "SAT"
  "SelfPowerSequenceMOD"
  "SetCover"
  "SetSplitting"
  "ShortestPath"
  "ShortestPathCountConstruction"
  "SkyscraperPuzzle"
  "SkyscraperSumPuzzle"
  "SlidingWindow"
  "SmallestBinaryMultiple"
  "SmallestCircle"
  "Sorting"
  "SpiralMatrix"
  "SpyNetwork"
  "StarBattle"
  "StirlingSecond"
  "StoneGame"
  "StringPartitionShuffle"
  "SubgraphIsomorphism"
  "SubsetSum"
  "SubsetSumSequence"
  "Sudoku"
  "Sum_DivisorNum"
  "SumGCD"
  "SumGCDWithIndividual"
  "SumLCM"
  "SumMOD"
  "SumProductDivisorNum"
  "SurvoPuzzle"
  "TakingPrimeGame"
  "TaskArrangement"
  "ThreeVertexCycleCounting"
  "TopologicalSort"
  "TreeCenter"
  "TreeColoring"
  "TreeDynamic_XORZeroPath"
  "TreeElimination_Expectation"
  "TwiddlePuzzle"
  "TwoSAT"
  "TwoSet_AllCoprime_Counting"
  "UndamagedSubmatrixCounting"
  "ValueDiminishingSelection"
  "Vertex_KCenter"
  "WarehouseConstruction"
  "WeightedBinaryTree"
  "WeightedLIS"
  "WhackAMole"
  "XorEquationCounting"
  "ZebraLogics"
)

python scripts/data/generate_super_rlvr_data.py \
    --task_list "${tasks[@]}" \
    --samples_per_task 1000 \
    --difficulty_levels 10
"""

import argparse
from open_instruct.VerifiableProblem.verifiable.problems import problem2class
from open_instruct.VerifiableProblem.verifiable.parameter_controllers import problem2controller
import random
import pandas as pd
from datasets import Dataset
import os
from tqdm import tqdm
import concurrent.futures

def process_fn(prompt, parameters_dict,task_name):

    data = {
        "dataset": f"verifiable_problem_z",
        "label": {"task_name": task_name, "parameters": parameters_dict},
        "messages": [{
            "role": "user",
            "content": prompt
        }],
    }
    return data

def generate_instance(task_name, seed, parameter):
    instance = problem2class[task_name]()
    instance.generator(seed, parameter)
    prompt = instance.prompt_generator()
    parameters_dict = instance.__dict__
    return process_fn(prompt, parameters_dict, task_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_list', nargs='+')
    parser.add_argument('--samples_per_task', type=int)
    parser.add_argument('--difficulty_levels', type=int)
    args = parser.parse_args()

    seed = 42
    update_difficulty_every = args.samples_per_task // args.difficulty_levels
    all_data = []

    for task_name in tqdm(args.task_list):
        parameter_controller = problem2controller[task_name]()
        task_data = []

        for i in tqdm(range(args.samples_per_task)):
            parameter = parameter_controller.get_parameter_list()[0]
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_instance, task_name, seed, parameter)
                try:
                    result = future.result(timeout=1)
                    task_data.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"Skipped instance for task {task_name} at seed {seed} due to timeout.")
                except Exception as e:
                    print(f"Error generating instance for task {task_name} at seed {seed}: {e}")
            seed += 1

            if (i + 1) % update_difficulty_every == 0:
                parameter_list = parameter_controller.update()

        all_data.extend(task_data)

    ds = Dataset.from_list(all_data)
    ds.push_to_hub("hamishivi/verifiable_problem_z")