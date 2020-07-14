/**
 * Copyright 2020
 * @file cycle_check_directed graph.cpp
 *
 * @brief BFS and DFS algorithms to check for cycle in a directed graph.
 *
 * @author Anmol3299
 * contact: mittalanmol22@gmail.com
 *
 */

#include <iostream>       // for std::cout
#include <queue>          // for std::queue
#include <stdexcept>      // for throwing errors
#include <type_traits>    // for std::remove_reference_t
#include <unordered_map>  // for std::unordered_map
#include <utility>        // for std::move
#include <vector>         // for std::vector
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <valarray>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ACCURACY 1e-10 /**< maximum accuracy limit */

/**
 * Implementation of non-weighted directed edge of a graph.
 *
 * The source vertex of the edge is labelled "src" and destination vertex is
 * labelled "dest".
 */
struct Edge {
    unsigned int src;
    unsigned int dest;

    Edge() = delete;
    ~Edge() = default;
    Edge(Edge&&) = default;
    Edge& operator=(Edge&&) = default;
    Edge(Edge const&) = default;
    Edge& operator=(Edge const&) = default;

    /** Set the source and destination of the vertex.
     *
     * @param source is the source vertex of the edge.
     * @param destination is the destination vertex of the edge.
     */
    Edge(unsigned int source, unsigned int destination)
        : src(source), dest(destination) {}
};

using AdjList = std::unordered_map<unsigned int, std::vector<unsigned int>>;

/**
 * Implementation of graph class.
 *
 * The graph will be represented using Adjacency List representation.
 * This class contains 2 data members "m_vertices" & "m_adjList" used to
 * represent the number of vertices and adjacency list of the graph
 * respectively. The vertices are labelled 0 - (m_vertices - 1).
 */
class Graph {
 public:
    Graph() : m_vertices(0), m_adjList({}) {}
    ~Graph() = default;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;
    Graph(Graph const&) = default;
    Graph& operator=(Graph const&) = default;

    /** Create a graph from vertices and adjacency list.
     *
     * @param vertices specify the number of vertices the graph would contain.
     * @param adjList is the adjacency list representation of graph.
     */
    Graph(unsigned int vertices, AdjList const& adjList)
        : m_vertices(vertices), m_adjList(adjList) {}

    /** Create a graph from vertices and adjacency list.
     *
     * @param vertices specify the number of vertices the graph would contain.
     * @param adjList is the adjacency list representation of graph.
     */
    Graph(unsigned int vertices, AdjList&& adjList)
        : m_vertices(std::move(vertices)), m_adjList(std::move(adjList)) {}

    /** Create a graph from vertices and a set of edges.
     *
     * Adjacency list of the graph would be created from the set of edges. If
     * the source or destination of any edge has a value greater or equal to
     * number of vertices, then it would throw a range_error.
     *
     * @param vertices specify the number of vertices the graph would contain.
     * @param edges is a vector of edges.
     */
    Graph(unsigned int vertices, std::vector<Edge> const& edges)
        : m_vertices(vertices) {
        for (auto const& edge : edges) {
            if (edge.src >= vertices || edge.dest >= vertices) {
                throw std::range_error(
                    "Either src or dest of edge out of range");
            }
            m_adjList[edge.src].emplace_back(edge.dest);
        }
    }

    /** Return a const reference of the adjacency list.
     *
     * @return const reference to the adjacency list
     */
    std::remove_reference_t<AdjList> const& getAdjList() const {
        return m_adjList;
    }

    /**
     * @return number of vertices in the graph.
     */
    std::remove_reference_t<unsigned int> const& getVertices() const {
        return m_vertices;
    }

    /** Add vertices in the graph.
     *
     * @param num is the number of vertices to be added. It adds 1 vertex by
     * default.
     *
     */
    void addVertices(unsigned int num = 1) { m_vertices += num; }

    /** Add an edge in the graph.
     *
     * @param edge that needs to be added.
     */
    void addEdge(Edge const& edge) {
        if (edge.src >= m_vertices || edge.dest >= m_vertices) {
            throw std::range_error("Either src or dest of edge out of range");
        }
        m_adjList[edge.src].emplace_back(edge.dest);
    }

    /** Add an Edge in the graph
     *
     * @param source is source vertex of the edge.
     * @param destination is the destination vertex of the edge.
     */
    void addEdge(unsigned int source, unsigned int destination) {
        if (source >= m_vertices || destination >= m_vertices) {
            throw std::range_error(
                "Either source or destination of edge out of range");
        }
        m_adjList[source].emplace_back(destination);
    }

 private:
    unsigned int m_vertices;
    AdjList m_adjList;
};

class CycleCheck {
 private:
    enum nodeStates : uint8_t { not_visited = 0, in_stack, visited };

    /** Helper function of "isCyclicDFS".
     *
     * @param adjList is the adjacency list representation of some graph.
     * @param state is the state of the nodes of the graph.
     * @param node is the node being evaluated.
     *
     * @return true if graph has a cycle, else false.
     */
    static bool isCyclicDFSHelper(AdjList const& adjList,
                                  std::vector<nodeStates>* state,
                                  unsigned int node) {
        // Add node "in_stack" state.
        (*state)[node] = in_stack;

        // If the node has children, then recursively visit all children of the
        // node.
        if (auto const& it = adjList.find(node); it != adjList.end()) {
            for (auto child : it->second) {
                // If state of child node is "not_visited", evaluate that child
                // for presence of cycle.
                if (auto state_of_child = (*state)[child];
                    state_of_child == not_visited) {
                    if (isCyclicDFSHelper(adjList, state, child)) {
                        return true;
                    }
                } else if (state_of_child == in_stack) {
                    // If child node was "in_stack", then that means that there
                    // is a cycle in the graph. Return true for presence of the
                    // cycle.
                    return true;
                }
            }
        }

        // Current node has been evaluated for the presence of cycle and had no
        // cycle. Mark current node as "visited".
        (*state)[node] = visited;
        // Return that current node didn't result in any cycles.
        return false;
    }

 public:
    /** Driver function to check if a graph has a cycle.
     *
     * This function uses DFS to check for cycle in the graph.
     *
     * @param graph which needs to be evaluated for the presence of cycle.
     * @return true if a cycle is detected, else false.
     */
    static bool isCyclicDFS(Graph const& graph) {
        /** State of the node.
         *
         * It is a vector of "nodeStates" which represents the state node is in.
         * It can take only 3 values: "not_visited", "in_stack", and "visited".
         *
         * Initially, all nodes are in "not_visited" state.
         */
        std::vector<nodeStates> state(graph.getVertices(), not_visited);

        // Start visiting each node.
        for (auto node = 0; node < graph.getVertices(); node++) {
            // If a node is not visited, only then check for presence of cycle.
            // There is no need to check for presence of cycle for a visited
            // node as it has already been checked for presence of cycle.
            if (state[node] == not_visited) {
                // Check for cycle.
                if (isCyclicDFSHelper(graph.getAdjList(), &state, node)) {
                    return true;
                }
            }
        }

        // All nodes have been safely traversed, that means there is no cycle in
        // the graph. Return false.
        return false;
    }

    /** Check if a graph has cycle or not.
     *
     * This function uses BFS to check if a graph is cyclic or not.
     *
     * @param graph which needs to be evaluated for the presence of cycle.
     * @return true if a cycle is detected, else false.
     */
    static bool isCyclicBFS(Graph const& graph) {
        AdjList graphAjdList = graph.getAdjList();

        std::vector<unsigned int> indegree(graph.getVertices(), 0);
        // Calculate the indegree i.e. the number of incident edges to the node.
        for (auto const& [parent, children] : graphAjdList) {
            for (auto const& child : children) {
                indegree[child]++;
            }
        }

        std::queue<unsigned int> can_be_solved;
        for (auto node = 0; node < graph.getVertices(); node++) {
            // If a node doesn't have any input edges, then that node will
            // definately not result in a cycle and can be visited safely.
            if (!indegree[node]) {
                can_be_solved.emplace(node);
            }
        }

        // Vertices that need to be traversed.
        auto remain = graph.getVertices();
        // While there are safe nodes that we can visit.
        while (!can_be_solved.empty()) {
            auto front = can_be_solved.front();
            // Visit the node.
            can_be_solved.pop();
            // Decrease number of nodes that need to be traversed.
            remain--;

            // Visit all the children of the visited node.
            if (auto it = graphAjdList.find(front); it != graphAjdList.end()) {
                for (auto child : it->second) {
                    // Check if we can visited the node safely.
                    if (--indegree[child] == 0) {
                        // if node can be visited safely, then add that node to
                        // the visit queue.
                        can_be_solved.emplace(child);
                    }
                }
            }
        }

        // If there are still nodes that we can't visit, then it means that
        // there is a cycle and return true, else return false.
        return !(remain == 0);
    }
};

/**
 * Main function.
 *//**
 * @file
 * \brief Compute all possible approximate roots of any given polynomial using
 * [Durand Kerner
 * algorithm](https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method)
 * \author [Krishna Vedala](https://github.com/kvedala)
 *
 * Test the algorithm online:
 * https://gist.github.com/kvedala/27f1b0b6502af935f6917673ec43bcd7
 *
 * Try the highly unstable Wilkinson's polynomial:
 * ```
 * ./numerical_methods/durand_kerner_roots 1 -210 20615 -1256850 53327946
 * -1672280820 40171771630 -756111184500 11310276995381 -135585182899530
 * 1307535010540395 -10142299865511450 63030812099294896 -311333643161390640
 * 1206647803780373360 -3599979517947607200 8037811822645051776
 * -12870931245150988800 13803759753640704000 -8752948036761600000
 * 2432902008176640000
 * ```
 * Sample implementation results to compute approximate roots of the equation
 * \f$x^4-1=0\f$:\n
 * <img
 * src="https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/docs/images/numerical_methods/durand_kerner_error.svg"
 * width="400" alt="Error evolution during root approximations computed every
 * iteration."/> <img
 * src="https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/docs/images/numerical_methods/durand_kerner_roots.svg"
 * width="400" alt="Roots evolution - shows the initial approximation of the
 * roots and their convergence to a final approximation along with the iterative
 * approximations" />
 */
/**
 * Evaluate the value of a polynomial with given coefficients
 * \param[in] coeffs coefficients of the polynomial
 * \param[in] x point at which to evaluate the polynomial
 * \returns \f$f(x)\f$
 **/
std::complex<double> poly_function(const std::valarray<double> &coeffs,
                                   std::complex<double> x) {
    double real = 0.f, imag = 0.f;
    int n;

    // #ifdef _OPENMP
    // #pragma omp target teams distribute reduction(+ : real, imag)
    // #endif
    for (n = 0; n < coeffs.size(); n++) {
        std::complex<double> tmp =
            coeffs[n] * std::pow(x, coeffs.size() - n - 1);
        real += tmp.real();
        imag += tmp.imag();
    }

    return std::complex<double>(real, imag);
}

/**
 * create a textual form of complex number
 * \param[in] x point at which to evaluate the polynomial
 * \returns pointer to converted string
 */
const char *complex_str(const std::complex<double> &x) {
#define MAX_BUFF_SIZE 50
    static char msg[MAX_BUFF_SIZE];

    std::snprintf(msg, MAX_BUFF_SIZE, "% 7.04g%+7.04gj", x.real(), x.imag());

    return msg;
}

/**
 * check for termination condition
 * \param[in] delta point at which to evaluate the polynomial
 * \returns `false` if termination not reached
 * \returns `true` if termination reached
 */
bool check_termination(long double delta) {
    static long double past_delta = INFINITY;
    if (std::abs(past_delta - delta) <= ACCURACY || delta < ACCURACY)
        return true;
    past_delta = delta;
    return false;
}

/**
 * Implements Durand Kerner iterative algorithm to compute all roots of a
 * polynomial.
 *
 * \param[in] coeffs coefficients of the polynomial
 * \param[out] roots the computed roots of the polynomial
 * \param[in] write_log flag whether to save the log file (default = `false`)
 * \returns pair of values - number of iterations taken and final accuracy
 * achieved
 */
std::pair<uint32_t, double> durand_kerner_algo(
    const std::valarray<double> &coeffs,
    std::valarray<std::complex<double>> *roots, bool write_log = false) {
    long double tol_condition = 1;
    uint32_t iter = 0;
    int n;
    std::ofstream log_file;

    if (write_log) {
        /*
         * store intermediate values to a CSV file
         */
        log_file.open("durand_kerner.log.csv");
        if (!log_file.is_open()) {
            perror("Unable to create a storage log file!");
            std::exit(EXIT_FAILURE);
        }
        log_file << "iter#,";

        for (n = 0; n < roots->size(); n++) log_file << "root_" << n << ",";

        log_file << "avg. correction";
        log_file << "\n0,";
        for (n = 0; n < roots->size(); n++)
            log_file << complex_str((*roots)[n]) << ",";
    }

    bool break_loop = false;
    while (!check_termination(tol_condition) && iter < INT16_MAX &&
           !break_loop) {
        tol_condition = 0;
        iter++;
        break_loop = false;

        if (log_file.is_open())
            log_file << "\n" << iter << ",";

#ifdef _OPENMP
#pragma omp parallel for shared(break_loop, tol_condition)
#endif
        for (n = 0; n < roots->size(); n++) {
            if (break_loop)
                continue;

            std::complex<double> numerator, denominator;
            numerator = poly_function(coeffs, (*roots)[n]);
            denominator = 1.0;
            for (int i = 0; i < roots->size(); i++)
                if (i != n)
                    denominator *= (*roots)[n] - (*roots)[i];

            std::complex<long double> delta = numerator / denominator;

            if (std::isnan(std::abs(delta)) || std::isinf(std::abs(delta))) {
                std::cerr << "\n\nOverflow/underrun error - got value = "
                          << std::abs(delta) << "\n";
                // return std::pair<uint32_t, double>(iter, tol_condition);
                break_loop = true;
            }

            (*roots)[n] -= delta;

#ifdef _OPENMP
#pragma omp critical
#endif
            tol_condition = std::max(tol_condition, std::abs(std::abs(delta)));
        }
        // tol_condition /= (degree - 1);

        if (break_loop)
            break;

        if (log_file.is_open()) {
            for (n = 0; n < roots->size(); n++)
                log_file << complex_str((*roots)[n]) << ",";
        }

#if defined(DEBUG) || !defined(NDEBUG)
        if (iter % 500 == 0) {
            std::cout << "Iter: " << iter << "\t";
            for (n = 0; n < roots->size(); n++)
                std::cout << "\t" << complex_str((*roots)[n]);
            std::cout << "\t\tabsolute average change: " << tol_condition
                      << "\n";
        }
#endif

        if (log_file.is_open())
            log_file << tol_condition;
    }

    return std::pair<uint32_t, long double>(iter, tol_condition);
}

/**
 * Self test the algorithm by checking the roots for \f$x^2+4=0\f$ to which the
 * roots are \f$0 \pm 2i\f$
 */
void test1() {
    const std::valarray<double> coeffs = {1, 0, 4};  // x^2 - 2 = 0
    std::valarray<std::complex<double>> roots(2);
    std::valarray<std::complex<double>> expected = {
        std::complex<double>(0., 2.),
        std::complex<double>(0., -2.)  // known expected roots
    };

    /* initialize root approximations with random values */
    for (int n = 0; n < roots.size(); n++) {
        roots[n] = std::complex<double>(std::rand() % 100, std::rand() % 100);
        roots[n] -= 50.f;
        roots[n] /= 25.f;
    }

    auto result = durand_kerner_algo(coeffs, &roots, false);

    for (int i = 0; i < roots.size(); i++) {
        // check if approximations are have < 0.1% error with one of the
        // expected roots
        bool err1 = false;
        for (int j = 0; j < roots.size(); j++)
            err1 |= std::abs(std::abs(roots[i] - expected[j])) < 1e-3;
        assert(err1);
    }

    std::cout << "Test 1 passed! - " << result.first << " iterations, "
              << result.second << " accuracy"
              << "\n";
}

/**
 * Self test the algorithm by checking the roots for \f$0.015625x^3-1=0\f$ to
 * which the roots are \f$(4+0i),\,(-2\pm3.464i)\f$
 */
void test2() {
    const std::valarray<double> coeffs = {// 0.015625 x^3 - 1 = 0
                                          1. / 64., 0., 0., -1.};
    std::valarray<std::complex<double>> roots(3);
    const std::valarray<std::complex<double>> expected = {
        std::complex<double>(4., 0.), std::complex<double>(-2., 3.46410162),
        std::complex<double>(-2., -3.46410162)  // known expected roots
    };

    /* initialize root approximations with random values */
    for (int n = 0; n < roots.size(); n++) {
        roots[n] = std::complex<double>(std::rand() % 100, std::rand() % 100);
        roots[n] -= 50.f;
        roots[n] /= 25.f;
    }

    auto result = durand_kerner_algo(coeffs, &roots, false);

    for (int i = 0; i < roots.size(); i++) {
        // check if approximations are have < 0.1% error with one of the
        // expected roots
        bool err1 = false;
        for (int j = 0; j < roots.size(); j++)
            err1 |= std::abs(std::abs(roots[i] - expected[j])) < 1e-3;
        assert(err1);
    }

    std::cout << "Test 2 passed! - " << result.first << " iterations, "
              << result.second << " accuracy"
              << "\n";
}

/***
 * Main function.
 * The comandline input arguments are taken as coeffiecients of a
 *polynomial. For example, this command
 * ```sh
 * ./durand_kerner_roots 1 0 -4
 * ```
 * will find roots of the polynomial \f$1\cdot x^2 + 0\cdot x^1 + (-4)=0\f$
 **/
int main(int argc, char **argv) {
    // Instantiate the graph.
    Graph g(7, std::vector<Edge>{{0, 1}, {1, 2}, {2, 0}, {2, 5}, {3, 5}});
    // Check for cycle using BFS method.
    std::cout << CycleCheck::isCyclicBFS(g) << '\n';

    // Check for cycle using DFS method.
    std::cout << CycleCheck::isCyclicDFS(g) << '\n';
        /* initialize random seed: */
    std::srand(std::time(nullptr));

    if (argc < 2) {
        test1();  // run tests when no input is provided
        test2();  // and skip tests when input polynomial is provided
        std::cout << "Please pass the coefficients of the polynomial as "
                     "commandline "
                     "arguments.\n";
        return 0;
    }

    int n, degree = argc - 1;              // detected polynomial degree
    std::valarray<double> coeffs(degree);  // create coefficiencts array

    // number of roots = degree - 1
    std::valarray<std::complex<double>> s0(degree - 1);

    std::cout << "Computing the roots for:\n\t";
    for (n = 0; n < degree; n++) {
        coeffs[n] = strtod(argv[n + 1], nullptr);
        if (n < degree - 1 && coeffs[n] != 0)
            std::cout << "(" << coeffs[n] << ") x^" << degree - n - 1 << " + ";
        else if (coeffs[n] != 0)
            std::cout << "(" << coeffs[n] << ") x^" << degree - n - 1
                      << " = 0\n";

        /* initialize root approximations with random values */
        if (n < degree - 1) {
            s0[n] = std::complex<double>(std::rand() % 100, std::rand() % 100);
            s0[n] -= 50.f;
            s0[n] /= 50.f;
        }
    }

    // numerical errors less when the first coefficient is "1"
    // hence, we normalize the first coefficient
    {
        double tmp = coeffs[0];
        coeffs /= tmp;
    }

    clock_t end_time, start_time = clock();
    auto result = durand_kerner_algo(coeffs, &s0, true);
    end_time = clock();

    std::cout << "\nIterations: " << result.first << "\n";
    for (n = 0; n < degree - 1; n++)
        std::cout << "\t" << complex_str(s0[n]) << "\n";
    std::cout << "absolute average change: " << result.second << "\n";
    std::cout << "Time taken: "
              << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC
              << " sec\n";

    return 0;
}
