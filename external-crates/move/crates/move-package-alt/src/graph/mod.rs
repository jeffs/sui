use std::{collections::BTreeMap, path::Path, time::Duration};

use petgraph::graph::{DiGraph, NodeIndex};

use crate::{
    dependency::PinnedDependencyInfo,
    errors::PackageResult,
    flavor::MoveFlavor,
    package::{EnvironmentName, Package, PackageName, PackagePath},
};

struct PackageGraph<F: MoveFlavor> {
    inner: DiGraph<Package<F>, PackageName>,
}

impl<F: MoveFlavor> PackageGraph<F> {
    /// Check to see whether the resolution graph in the lockfile inside `path` is up-to-date (i.e.
    /// whether any of the manifests digests are out of date). If the resolution graph is
    /// up-to-date, it is returned. Otherwise a new resolution graph is constructed by traversing
    /// the manifest files (see [load_from_manifests] for details).
    pub async fn load(path: &PackagePath) -> PackageResult<Self> {
        if let Some(graph) = Self::load_from_lockfile(path).await? {
            Ok(graph)
        } else {
            Self::load_from_manifests(path).await
        }
    }

    /// Construct a [PackageGraph] by pinning and fetching all transitive dependencies from the
    /// manifest in `path`.
    pub async fn load_from_manifests(path: &PackagePath) -> PackageResult<Self> {
        let mut result = Self {
            inner: DiGraph::new(),
        };

        // .add_transitive_dependencies(&PinnedDependencyInfo::Root, &mut BTreeMap::new())
        //     .await?;

        Ok(result)
    }

    /// Load a [PackageGraph] from the lockfile at `path`. Returns [None] if the contents of the
    /// lockfile are out of date (i.e. if the lockfile doesn't exist or the manifest digests don't
    /// match).
    async fn load_from_lockfile(path: &PackagePath) -> PackageResult<Option<Self>> {
        todo!()
    }
}

/// Add [package] and all of its transitive dependencies to `self.inner`.
async fn add_transitive_dependencies<F: MoveFlavor>(
    dep: &PinnedDependencyInfo<F>,
    cache: &mut tokio::sync::Mutex<(
        DiGraph<Package<F>, (EnvironmentName, PackageName)>,
        BTreeMap<PackagePath, NodeIndex>,
    )>,
) -> PackageResult<NodeIndex> {
    // lock the graph and cache
    // check cache - if index in there, return it
    // otherwise fetch the dependency
    // add node to the graph and the cache
    // release the lock
    // recursively add dependencies and add edges
    let index = {
        let path = dep.fetch();
        let (graph, visited) = cache.get_mut();
        let package = match visited.get(&path) {
            Some(index) => return Ok(*index),
            None => Package::load(dep.clone()).await?,
        };
        let index = graph.add_node(package);
        visited.insert(path, index);
        index
    };

    for (env, name, dep) in package.pinned_direct_dependencies() {
        let index = add_transitive_dependencies(dep, cache);
    }

    todo!()
}
