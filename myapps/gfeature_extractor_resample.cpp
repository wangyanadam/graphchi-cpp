/**
 * @file
 * @author Yan Wang <yw298@cs.rutgers.edu> 
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Yan Wang / Rutgers University]
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 
 *
 * @section DESCRIPTION
 *
 * Graph feature extractor for a friendship graph with (a) edge weight being
 * binary, (b) undirected edges, and (c) vertices coming from a single domain.
 * The feature extraction follows the similar procedure as the ReFex, which
 * consists of computing neighborhood features at first and then iteratively
 * computing recursive features based on the neighborhood features.
 */

#include <string>
#include <set>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "graphchi_basic_includes.hpp"

using namespace graphchi;
using std::sqrt;
using std::cout;
using std::endl;
using std::cerr;
using std::clog;

/*
 * Number of extracted features (>=3).
 * */
const int NumFeatures = 10;

/*
 * Number of sampling rounds for computing the neighborhood features.
 */
const int NumSampleRounds = 1000;

/*
 * Size of the random neighborhood sample.
 */
const int NumRndNbs = 5;

// Feature value type
typedef float FValue;

// Vertex Data Type
struct VertexDataType {

	// Currently propagated feature's ID
	int pass_fid;

	// ID of the feature waiting for being filled
	int fill_fid;

	// Feature vector
	FValue fvals[NumFeatures];

	VertexDataType() :
			pass_fid(-1), fill_fid(-1), fvals { 0 } {
	}
};

// Edge Data Type
struct EdgeDataType {
	FValue v1_fval;
	int v1_fid;

	FValue v2_fval;
	int v2_fid;

	vid_t v1_nbs[NumRndNbs];
	vid_t v2_nbs[NumRndNbs];

	EdgeDataType() :
			v1_fval(-1), v1_fid(-1), v2_fval(-1), v2_fid(-1), v1_nbs { 0 }, v2_nbs {
					0 } {
	}
};

// Output file
FILE *outfile = NULL;

// GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
// class. The main logic is usually in the update function.
struct GFeatureExtractor: public GraphChiProgram<VertexDataType, EdgeDataType> {

	/*
	 * Neighborhood features:
	 * a.1) Degree of v,
	 * a.2) Clustering coefficient = triangle-count(v) / (deg(v) * (deg(v) - 1)),
	 * a.3) Cut of the 1-egonet(v) = #{edges with only one end in the 1-egonet(v)}.
	 *
	 * Recursive features:
	 * Recursively generated based on the above 3 neighbor features.
	 *
	 * Extracting features a.2) and a.3):
	 * On the message passing iteration, the vertex will send a
	 * random sample of its neighborhood and pass the sample to
	 * all the neighbors. On the message aggregation iteration,
	 * the vertex will compute the features approximately based
	 * on the received neighborhood samples from its neighbors.
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &v,
			graphchi_context &ginfo) {

		int num_edges = v.num_edges();

		if (ginfo.iteration == 0) {

			// Init vertex data
			v.dataptr->fvals[0] = num_edges;
			for (int i = 1; i < NumFeatures; i++) {
				v.dataptr->fvals[i] = 0;
			}
			v.dataptr->pass_fid = 0;
			v.dataptr->fill_fid = 4;

			// Init edge data
			for (int i = 0; i < num_edges; i++) {
				graphchi_edge<EdgeDataType> *cur_edge = v.edge(i);

				int *fid;
				FValue *fval;
				vid_t *msg_nbs;

				if (cur_edge->vertexid < v.vertexid) {
					fid = &(cur_edge->data_ptr->v1_fid);
					fval = &(cur_edge->data_ptr->v1_fval);
					msg_nbs = cur_edge->data_ptr->v1_nbs;
				} else {
					fid = &(cur_edge->data_ptr->v2_fid);
					fval = &(cur_edge->data_ptr->v2_fval);
					msg_nbs = cur_edge->data_ptr->v2_nbs;
				}

				*fid = 0;
				*fval = num_edges;

				for (int j = 0; j < NumRndNbs; j++) {
					msg_nbs[j] =
							v.edge((int) (std::abs(std::rand()) % num_edges))->vertexid;
				}
			}

		} else if (ginfo.iteration <= NumSampleRounds) {

			// Aggregating messages
			std::set<vid_t> nb_ids;
			for (int i = 0; i < num_edges; i++) {
				nb_ids.insert(v.edge(i)->vertexid);
			}
			nb_ids.insert(v.vertexid);

			double tri_count = 0;
			double oedge_count = 0;
			for (int i = 0; i < num_edges; i++) {
				graphchi_edge<EdgeDataType> *cur_edge = v.edge(i);

				FValue *fval;
				vid_t *msg_nbs;

				if (cur_edge->vertexid > v.vertexid) {
					fval = &(cur_edge->data_ptr->v1_fval);
					msg_nbs = cur_edge->data_ptr->v1_nbs;
				} else {
					fval = &(cur_edge->data_ptr->v2_fval);
					msg_nbs = cur_edge->data_ptr->v2_nbs;
				}

				double in_net_ratio = 0;
				double out_net_ratio = 0;
				for (int j = 0; j < NumRndNbs; j++) {
					if (nb_ids.find(msg_nbs[j]) != nb_ids.end()) {
						if (msg_nbs[j] != v.vertexid) {
							in_net_ratio++;
						}
					} else {
						out_net_ratio++;
					}
				}

				in_net_ratio = in_net_ratio / NumRndNbs;
				tri_count += in_net_ratio * (*fval);

				out_net_ratio = out_net_ratio / NumRndNbs;
				oedge_count += out_net_ratio * (*fval);
			}

			// Updating ratio
			double ur = 1.0 / ginfo.iteration;

			// Updating f1, i.e. clustering coefficient
			if (num_edges > 1) {
				v.dataptr->fvals[1] = (1 - ur) * v.dataptr->fvals[1]
						+ ur * (tri_count / (num_edges * (num_edges - 1)));
			}

			// Updating f2, i.e. in-egonet-edge count
			tri_count = tri_count / 2;
			v.dataptr->fvals[2] = (1 - ur) * v.dataptr->fvals[2]
					+ ur * tri_count;

			// Updating f3, i.e. cut ratio
			if (num_edges + tri_count + oedge_count > 0) {
				v.dataptr->fvals[3] =
						(1 - ur) * v.dataptr->fvals[3]
								+ ur
										* (oedge_count
												/ (num_edges + tri_count
														+ oedge_count));
			}

			// Updating f4, i.e. outside-edge count
			v.dataptr->fvals[4] = (1 - ur) * v.dataptr->fvals[4]
					+ ur * oedge_count;

			// Propagating messages
			for (int i = 0; i < num_edges; i++) {
				graphchi_edge<EdgeDataType> *cur_edge = v.edge(i);

				vid_t *msg_nbs;
				if (cur_edge->vertexid < v.vertexid) {
					msg_nbs = cur_edge->data_ptr->v1_nbs;
				} else {
					msg_nbs = cur_edge->data_ptr->v2_nbs;
				}

				for (int j = 0; j < NumRndNbs; j++) {
					msg_nbs[j] =
							v.edge((int) (std::abs(std::rand()) % num_edges))->vertexid;
				}
			}

		} else {

			// Computing recursive features
			if (((ginfo.iteration - NumSampleRounds) % 2) == 1) {

				// Propagating messages
				for (int i = 0; i < num_edges; i++) {
					graphchi_edge<EdgeDataType> *cur_edge = v.edge(i);

					int *fid;
					FValue *fval;

					if (cur_edge->vertexid < v.vertexid) {
						fid = &(cur_edge->data_ptr->v1_fid);
						fval = &(cur_edge->data_ptr->v1_fval);
					} else {
						fid = &(cur_edge->data_ptr->v2_fid);
						fval = &(cur_edge->data_ptr->v2_fval);
					}

					*fid = v.dataptr->pass_fid;
					*fval = v.dataptr->fvals[*fid];
				}

				v.dataptr->pass_fid++;

			} else {

				// Aggregating messages
				if (v.dataptr->fill_fid + 1 >= NumFeatures) {
					ginfo.set_last_iteration(ginfo.iteration);
				} else {
					FValue mf = 0;
					FValue vf = 0;
					for (int i = 0; i < num_edges; i++) {
						graphchi_edge<EdgeDataType> *cur_edge = v.edge(i);

						FValue *fval;
						if (cur_edge->vertexid > v.vertexid) {
							fval = &(cur_edge->data_ptr->v1_fval);
						} else {
							fval = &(cur_edge->data_ptr->v2_fval);
						}

						mf += (*fval);
						vf += ((*fval) * (*fval));
					}

					mf = mf / num_edges;
					vf = vf / num_edges - mf * mf;

					// According to the definition of empirical standard deviation
					if (vf != 0) {
						vf = vf * num_edges / (num_edges - 1);
					}

					v.dataptr->fill_fid++;
					v.dataptr->fvals[v.dataptr->fill_fid] = mf;

					if (v.dataptr->fill_fid + 1 >= NumFeatures) {
						ginfo.set_last_iteration(ginfo.iteration);
					} else {
						v.dataptr->fill_fid++;
						v.dataptr->fvals[v.dataptr->fill_fid] = sqrt(vf);

						if (v.dataptr->fill_fid + 1 >= NumFeatures) {
							ginfo.set_last_iteration(ginfo.iteration);
						}
					}
				}
			}
		}
	}
	/**
	 * Called before an iteration starts.
	 */
	void before_iteration(int iteration, graphchi_context &ginfo) {
	}

	/**
	 * Called after an iteration has finished.
	 */
	void after_iteration(int iteration, graphchi_context &ginfo) {
	}

	/**
	 * Called before an execution interval is started.
	 */
	void before_exec_interval(vid_t window_st, vid_t window_en,
			graphchi_context &gcontext) {
	}

	/**
	 * Called after an execution interval has finished.
	 */
	void after_exec_interval(vid_t window_st, vid_t window_en,
			graphchi_context &ginfo) {
	}

}
;

/*To-Do*/
// If the graph can be fully loaded into the memory, then the following
// in-memory vertex update subroutine will be called rather than the
// above ordinary one.
struct GFeatureExtractorInmem: public GraphChiProgram<VertexDataType,
		EdgeDataType> {

	std::vector<VertexDataType> vlist;
	GFeatureExtractorInmem(int nv) :
			vlist(nv) {
	}

	/**
	 *  Vertex update function.
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex,
			graphchi_context &ginfo) {

		if (ginfo.iteration == 0) {
			/* On first iteration, initialize vertex (and its edges). This is usually required, because
			 on each run, GraphChi will modify the data files. To start from scratch, it is easiest
			 do initialize the program in code. Alternatively, you can keep a copy of initial data files. */
			// vertex.set_data(init_value);
		} else {
			/* Do computation */

			/* Loop over in-edges (example) */
			for (int i = 0; i < vertex.num_inedges(); i++) {
				// Do something
				//    value += vertex.inedge(i).get_data();
			}

			/* Loop over out-edges (example) */
			for (int i = 0; i < vertex.num_outedges(); i++) {
				// Do something
				// vertex.outedge(i).set_data(x)
			}

			/* Loop over all edges (ignore direction) */
			for (int i = 0; i < vertex.num_edges(); i++) {
				// vertex.edge(i).get_data()
			}

			// v.set_data(new_value);
		}
	}

	/**
	 * Called before an iteration starts.
	 */
	void before_iteration(int iteration, graphchi_context &ginfo) {
	}

	/**
	 * Called after an iteration has finished.
	 */
	void after_iteration(int iteration, graphchi_context &ginfo) {
	}

	/**
	 * Called before an execution interval is started.
	 */
	void before_exec_interval(vid_t window_st, vid_t window_en,
			graphchi_context &ginfo) {
	}

	/**
	 * Called after an execution interval has finished.
	 */
	void after_exec_interval(vid_t window_st, vid_t window_en,
			graphchi_context &ginfo) {
	}
};

/* class for output the feature values for each node (optional) */
class OutputVertexCallback: public VCallback<VertexDataType> {
public:
	/* print node id and then the feature values */
	virtual void callback(vid_t vertex_id, VertexDataType &value) {
		fprintf(outfile, "%u", vertex_id);
		for (int i = 0; i < NumFeatures; i++) {
			fprintf(outfile, "\t%f", value.fvals[i]);
		}
		fprintf(outfile, "\n");
	}
};

int main(int argc, const char ** argv) {
	/* GraphChi initialization will read the command line
	 arguments and the configuration file. */
	graphchi_init(argc, argv);

// Initialize random number generator
	std::srand(time(NULL));

	/* Metrics object for keeping track of performance counters
	 and other information. Currently required. */
	metrics m("gfeature_extractor");

	/* Basic arguments for application */
	std::string infilename = get_option_string("input"); // Base filename
	std::string outfilename = get_option_string("output"); // Output filename
	int niters = get_option_int("niters", 1); // Number of iterations
	bool scheduler = get_option_int("scheduler", 0); // Whether to use selective schedulig

	/* Detect the number of shards or preprocess an input to create them */
	int nshards = convert_if_notexists<EdgeDataType>(infilename,
			get_option_string("nshards", "auto"));

	/* Run */
	graphchi_engine<VertexDataType, EdgeDataType> engine(infilename, nshards,
			scheduler, m);

	engine.set_modifies_outedges(false);
	engine.set_modifies_inedges(false);
	engine.set_save_edgesfiles_after_inmemmode(true);

	GFeatureExtractor program;
	engine.run(program, niters);

	/*To-Do*/
//	bool inmemmode = engine.num_vertices() * sizeof(EdgeDataType)
//			< (size_t) engine.get_membudget_mb() * 1024L * 1024L;
//
//	if (inmemmode) {
//		logstream(LOG_INFO)
//				<< "Running GFeature by holding vertices in-memory mode!"
//				<< std::endl;
//		GFeatureExtractorInmem program(engine.num_vertices());
//		engine.run(program, niters);
//	} else {
//		GFeatureExtractor program;
//		engine.run(program, niters);
//	}
	/* Output */
	outfile = fopen(outfilename.c_str(), "w");
	if (!outfile)
		logstream(LOG_FATAL) << "Failed to open file: " << outfilename
				<< std::endl;

	OutputVertexCallback callback;
	foreach_vertices<VertexDataType>(infilename, 1, engine.num_vertices(),
			callback);
	fclose(outfile);

	/* Report execution metrics */
	metrics_report(m);
	return 0;
}
