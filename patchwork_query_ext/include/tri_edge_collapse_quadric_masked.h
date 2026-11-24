#ifndef TRI_EDGE_COLLAPSE_QUADRIC_MASKED
#define TRI_EDGE_COLLAPSE_QUADRIC_MASKED

#include "eigen_define.h"
#include <Eigen/Core>
#include <unordered_set>
#include <vector>

#include "tri_edge_collapse_quadric_impl.h"
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/complex.h>
#include <vcg/math/quadric.h>

class MyFace;
class MyEdge;
class MyVertex;

struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>::AsVertexType,
                                           vcg::Use<MyEdge>::AsEdgeType,
                                           vcg::Use<MyFace>::AsFaceType> {};

class MyVertex
    : public vcg::Vertex<MyUsedTypes, vcg::vertex::VFAdj, vcg::vertex::Coord3f,
                         vcg::vertex::Mark, vcg::vertex::Qualityf,
                         vcg::vertex::BitFlags> {
public:
  vcg::math::Quadric<double> &Qd() { return q; }
  int &Level() { return level; }

private:
  vcg::math::Quadric<double> q;
  int level;
};

class MyEdge : public vcg::Edge<MyUsedTypes> {};

typedef vcg::tri::BasicVertexPair<MyVertex> VertexPair;

class MyFace
    : public vcg::Face<MyUsedTypes, vcg::face::VertexRef, vcg::face::FFAdj,
                       vcg::face::VFAdj, vcg::face::Normal3f,
                       vcg::face::Qualityf, vcg::face::BitFlags> {
public:
  int &A() { return a; }
  int &B() { return b; }
  int cA() const { return a; }
  int cB() const { return b; }

private:
  int a, b;
};
class MyMesh
    : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace>> {};

class MyTriEdgeCollapse : public vcg::tri::TriEdgeCollapseQuadric<
                              MyMesh, VertexPair, MyTriEdgeCollapse,
                              vcg::tri::QInfoStandard<MyVertex>> {
public:
  typedef vcg::tri::TriEdgeCollapseQuadric<
      MyMesh, VertexPair, MyTriEdgeCollapse, vcg::tri::QInfoStandard<MyVertex>>
      TECQ;
  typedef MyMesh::VertexType::EdgeType EdgeType;
  inline MyTriEdgeCollapse(const VertexPair &p, int i,
                           vcg::BaseParameterClass *pp)
      : TECQ(p, i, pp) {}
};

inline std::uint64_t PackUint64(const std::pair<int, int> &p) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.first))
          << 32) |
         static_cast<std::uint32_t>(p.second);
}

inline bool IsUnmarked(MyFace *f) { return (f->cA() == -1) && (f->cB() == -1); }

inline bool IsMarked(MyFace *f) { return !IsUnmarked(f); }

inline bool IsCross(MyFace *f, MyFace *f_oppo) {
  return f->cA() != -1 && f->cB() != -1 && f_oppo->cA() != -1 &&
         f_oppo->cB() != -1 &&
         (f->cA() != f_oppo->cA() || f->cB() != f_oppo->cB());
}

inline bool IsAny(MyFace *f, MyFace *f_oppo) {
  return f->cA() != -1 || f->cB() != -1 || f_oppo->cA() != -1 ||
         f_oppo->cB() != -1;
}

inline std::pair<MyFace *, int> OppositeFace(MyFace *f, int e) {
  MyVertex *v0 = f->V0(e), *v1 = f->V1(e);
  vcg::face::VFIterator<MyFace> x(v0);
  for (; !x.End(); ++x) {
    if (x.F() == f)
      continue;
    if (x.V1() == v1)
      return {x.F(), x.I()};
    if (x.V2() == v1)
      return {x.F(), (x.I() + 2) % 3};
  }
  return {f, e};
}

template <typename... Ts> struct CompareFunc {
  bool operator()(const std::tuple<int, Ts...> &a,
                  const std::tuple<int, Ts...> &b) const {
    return std::get<0>(a) < std::get<0>(b);
  }
};
template <typename... Ts>
using PriorityQueue = std::priority_queue<std::tuple<int, Ts...>,
                                          std::vector<std::tuple<int, Ts...>>,
                                          CompareFunc<Ts...>>;

template <typename Populate> void DoPass(MyMesh &m, Populate populate) {
  PriorityQueue<MyVertex *, MyVertex *> pq;
  size_t last_size = 0;
  while (true) {
    populate(pq);
    if (pq.size() == last_size)
      break;
    std::cout << "Heap size: " << pq.size() << std::endl;
    last_size = pq.size();
    while (!pq.empty()) {
      auto [level, v0, v1] = pq.top();
      pq.pop();
      if (!v0->IsD() && !v1->IsD()) {
        if (v0->Level() < v1->Level()) {
          VertexPair pair(v0, v1);
          // v0 will be deleted and v1 will survive, so higher level preserves
          vcg::tri::EdgeCollapser<MyMesh, VertexPair>::Do(m, pair, v1->P());
        } else {
          VertexPair pair(v1, v0);
          vcg::tri::EdgeCollapser<MyMesh, VertexPair>::Do(
              m, pair,
              v0->Level() == v1->Level() ? 0.5f * (v0->P() + v1->P())
                                         : v0->P());
        }
      }
    }
  }
}

inline void SharedCollapse(MyMesh &mesh) {
  DoPass(mesh, [&](PriorityQueue<MyVertex *, MyVertex *> &priority_queue) {
    for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
      if (vi->Level() == 0 || vi->IsD()) {
        continue;
      }

      vcg::face::VFIterator<MyFace> x;
      for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
        x.V1()->ClearV();
        x.V2()->ClearV();
      }
      for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
        MyFace *f = x.F();
        if (f->IsD()) {
          continue;
        }
        int e = x.I();
        float alpha = vcg::face::WedgeAngleRad(*f, (e + 2) % 3);

        if ((x.V0() < x.V1()) && !x.V1()->IsV()) {
          auto [f_oppo, e_oppo] = OppositeFace(f, e);
          float beta = vcg::face::WedgeAngleRad(*f_oppo, (e_oppo + 2) % 3);
          if (IsCross(f, f_oppo) && (alpha < M_PI / 6.0 && beta < M_PI / 6.0)) {
            x.V1()->SetV();
            priority_queue.emplace(std::max(x.V0()->Level(), x.V1()->Level()),
                                   x.V0(), x.V1());
          }
        }
        if ((x.V0() < x.V2()) && !x.V2()->IsV()) {
          auto [f_oppo, e_oppo] = OppositeFace(f, e);
          float beta = vcg::face::WedgeAngleRad(*f_oppo, (e_oppo + 2) % 3);
          if (IsCross(f, f_oppo) && (alpha < M_PI / 6.0 && beta < M_PI / 6.0)) {
            x.V2()->SetV();
            priority_queue.emplace(std::max(x.V0()->Level(), x.V2()->Level()),
                                   x.V0(), x.V2());
          }
        }
      }
    }
  });
}

// Split obtuse
struct SplitPair {
  size_t fid_marked;
  int vid_marked;
  size_t fid_unmarked;
  int vid_unmarked;
};
inline int ObtuseSplit(MyMesh &mesh, float obtuse_thr = 0.667 * M_PI) {
  std::vector<SplitPair> split_pair_vec;
  for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
    vcg::face::VFIterator<MyFace> x;
    for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
      x.F()->ClearV();
    }
    for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
      MyFace *f = x.F();
      // Only consider marked faces
      if (f->IsD() || f->IsV() || IsUnmarked(f)) {
        continue;
      }

      // Identify obtuse vid
      int split_vid = -1;
      for (int offset = 0; offset < 3; offset++) {
        int vid = (x.I() + offset) % 3;
        if (vcg::face::WedgeAngleRad(*f, vid) > obtuse_thr) {
          split_vid = vid;
        }
      }

      if (split_vid == -1) {
        continue;
      }

      int split_eid = (split_vid + 1) % 3;
      auto [f_oppo, split_eid_opp] = OppositeFace(f, split_eid);
      int split_vid_oppo = (split_eid_opp + 2) % 3;

      if (!f_oppo->IsD() && IsUnmarked(f_oppo) && !f_oppo->IsV()) {
        f->SetV();
        f_oppo->SetV();
        // I think I need to save index, as the reallocation could happen
        split_pair_vec.push_back({vcg::tri::Index(mesh, f), split_vid,
                                  vcg::tri::Index(mesh, f_oppo),
                                  split_vid_oppo});
      }
    }
  }

  // Allocate additions
  auto vi =
      vcg::tri::Allocator<MyMesh>::AddVertices(mesh, split_pair_vec.size());
  auto fi =
      vcg::tri::Allocator<MyMesh>::AddFaces(mesh, 2 * split_pair_vec.size());

  for (auto split : split_pair_vec) {
    auto &f_marked = mesh.face[split.fid_marked];
    auto &f_unmarked = mesh.face[split.fid_unmarked];

    auto v0 = f_marked.V(split.vid_marked);
    auto v1 = f_marked.V((split.vid_marked + 1) % 3);
    auto v2 = f_marked.V((split.vid_marked + 2) % 3);

    // TODO: Update VFAdj here
    float theta =
        vcg::face::WedgeAngleRad(f_marked, (split.vid_marked + 1) % 3);
    vi->P() = (v0->cP() - v1->cP()).Norm() * std::cos(theta) *
                  (v2->P() - v1->cP()).normalized() +
              v1->cP();
    vi->Level() = std::max(v1->Level(), v2->Level());

    fi->V(0) = f_marked.V(split.vid_marked);
    fi->V(1) = &(*vi);
    fi->V(2) = f_marked.V((split.vid_marked + 2) % 3);
    fi->A() = f_marked.cA();
    fi->B() = f_marked.cB();
    fi++;
    f_marked.V((split.vid_marked + 2) % 3) = &(*vi);

    fi->V(0) = f_unmarked.V(split.vid_unmarked);
    fi->V(1) = &(*vi);
    fi->V(2) = f_unmarked.V((split.vid_unmarked + 2) % 3);
    fi->A() = f_unmarked.cA();
    fi->B() = f_unmarked.cB();
    fi++;
    f_unmarked.V((split.vid_unmarked + 2) % 3) = &(*vi);

    vi++;
  }
  vcg::tri::UpdateTopology<MyMesh>::VertexFace(mesh);
  return split_pair_vec.size();
}

inline void SilverCollapse(MyMesh &mesh) {
  DoPass(mesh, [&](PriorityQueue<MyVertex *, MyVertex *> &priority_queue) {
    for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
      if (vi->Level() == 0 || vi->IsD()) {
        continue;
      }

      vcg::face::VFIterator<MyFace> x;
      for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
        x.V1()->ClearV();
        x.V2()->ClearV();
      }
      for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
        MyFace *f = x.F();
        if (f->IsD() || IsUnmarked(f)) {
          continue;
        }
        int e = x.I();

        float el_0 = (x.V0()->P() - x.V1()->P()).Norm();
        float el_1 = (x.V1()->P() - x.V2()->P()).Norm();
        float el_2 = (x.V2()->P() - x.V0()->P()).Norm();

        if ((x.V0() < x.V1()) && !x.V1()->IsV()) {
          if (el_0 < 0.2 * std::min(el_1, el_2)) {
            x.V1()->SetV();
            priority_queue.emplace(std::max(x.V0()->Level(), x.V1()->Level()),
                                   x.V0(), x.V1());
          }
        }
        if ((x.V0() < x.V2()) && !x.V2()->IsV()) {
          if (el_2 < 0.2 * std::min(el_1, el_0)) {
            x.V2()->SetV();
            priority_queue.emplace(std::max(x.V0()->Level(), x.V2()->Level()),
                                   x.V0(), x.V2());
          }
        }
      }
    }
  });
}

inline void SilverCollapse2(MyMesh &mesh) {
  DoPass(mesh, [&](PriorityQueue<MyVertex *, MyVertex *> &priority_queue) {
    for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
      if (vi->Level() == 0 || vi->IsD()) {
        continue;
      }

      vcg::face::VFIterator<MyFace> x;
      for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
        x.V1()->ClearV();
        x.V2()->ClearV();
      }
      for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
        MyFace *f = x.F();
        if (f->IsD() || IsUnmarked(f)) {
          continue;
        }
        int e = x.I();

        // 02
        float ang_1 = vcg::face::WedgeAngleRad(*f, (e + 1) % 3);
        // 01
        float ang_2 = vcg::face::WedgeAngleRad(*f, (e + 2) % 3);

        if ((x.V0() < x.V1()) && !x.V1()->IsV()) {
          if (ang_2 < M_PI / 12 && ang_2 < ang_1) {
            x.V1()->SetV();
            priority_queue.emplace(std::max(x.V0()->Level(), x.V1()->Level()),
                                   x.V0(), x.V1());
          }
        }
        if ((x.V0() < x.V2()) && !x.V2()->IsV()) {
          if (ang_1 < M_PI / 12 && ang_1 < ang_2) {
            x.V2()->SetV();
            priority_queue.emplace(std::max(x.V0()->Level(), x.V2()->Level()),
                                   x.V0(), x.V2());
          }
        }
      }
    }
  });
}

inline void tri_edge_collapse_quadric_masked(
    Eigen::Ref<const RowMatrixXf> V, Eigen::Ref<const RowMatrixXi> F,
    Eigen::Ref<const Eigen::VectorXi> F_label,
    Eigen::Ref<const Eigen::VectorXi> F_label2, RowMatrixXf &U, RowMatrixXi &G,
    Eigen::VectorXi &C1, Eigen::VectorXi &C2, int max_iter = 1000) {
  std::vector<vcg::Point3f> coordVec;
  coordVec.resize(V.rows());
  std::vector<vcg::Point3i> indexVec;
  indexVec.resize(F.rows());

  for (int vid = 0; vid < V.rows(); vid++) {
    coordVec[vid] = {V(vid, 0), V(vid, 1), V(vid, 2)};
  }
  for (int fid = 0; fid < F.rows(); fid++) {
    indexVec[fid] = {F(fid, 0), F(fid, 1), F(fid, 2)};
  }

  MyMesh mesh;
  vcg::tri::BuildMeshFromCoordVectorIndexVector(mesh, coordVec, indexVec);

  for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
    (*fi).A() = F_label(fi - mesh.face.begin());
    (*fi).B() = F_label2(fi - mesh.face.begin());
  }

  int null = vcg::tri::Clean<MyMesh>::RemoveFaceOutOfRangeArea(mesh, 0);
  int dup = vcg::tri::Clean<MyMesh>::RemoveDuplicateVertex(mesh);
  int unref = vcg::tri::Clean<MyMesh>::RemoveUnreferencedVertex(mesh);
  printf("Pre-cleaning: Removed %i null faces, %i duplicate and %i "
         "unreferenced vertices from mesh \n",
         null, dup, unref);

  vcg::tri::UpdateTopology<MyMesh>::VertexFace(mesh);

  // Loop all vertices to get count
  for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
    vcg::face::VFIterator<MyFace> x;
    // Loop all one ring
    std::unordered_set<std::uint64_t> label_set;
    for (x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F() != 0; ++x) {
      MyFace *f = x.F();
      if (IsMarked(f)) {
        label_set.insert(PackUint64({f->cA(), f->cB()}));
      }
    }
    vi->Level() = label_set.size();
  }

  // Core algorithm
  // Pass 1: Shared collapse
  SharedCollapse(mesh);

  // Pass 2: Silver removal
  SilverCollapse2(mesh);

  null = vcg::tri::Clean<MyMesh>::RemoveFaceOutOfRangeArea(mesh, 0);
  dup = vcg::tri::Clean<MyMesh>::RemoveDuplicateVertex(mesh);
  unref = vcg::tri::Clean<MyMesh>::RemoveUnreferencedVertex(mesh);
  printf("Post-cleaning: Removed %i null faces, %i duplicate and %i "
         "unreferenced vertices from mesh \n",
         null, dup, unref);

  std::vector<int> vid_vec(mesh.vert.size());

  U.resize(mesh.VN(), 3);
  G.resize(mesh.FN(), 3);
  C1.resize(mesh.FN());
  C2.resize(mesh.FN());

  int v_count = 0, f_count = 0;
  for (auto vi = mesh.vert.begin(); vi != mesh.vert.end(); ++vi) {
    if (!(*vi).IsD()) {
      vid_vec[vi - mesh.vert.begin()] = v_count;
      const auto &coord = (*vi).cP();
      U.row(v_count) << coord[0], coord[1], coord[2];
      v_count++;
    }
  }

  for (auto fi = mesh.face.begin(); fi != mesh.face.end(); ++fi) {
    if (!(*fi).IsD()) {
      Eigen::RowVector3i f;
      for (int k = 0; k < (*fi).VN(); k++) {
        f(k) = vid_vec[vcg::tri::Index(mesh, (*fi).V(k))];
      }
      G.row(f_count) = f;
      C1(f_count) = (*fi).cA();
      C2(f_count) = (*fi).cB();
      f_count++;
    }
  }
}

#endif // TRI_EDGE_COLLAPSE_QUADRIC_MASKED