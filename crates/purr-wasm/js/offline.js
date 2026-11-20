import { env } from "@huggingface/transformers";

const CACHE_NAME = "transformers-cache";
const BRANCH_URL = /^(https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/)main(\/.+)$/;

const revisions = new Map();

export function pinRevision(repo, revision) {
    revisions.set(repo, revision);
}

// transformers.js asks for some files on `main` before it honours the pinned revision
function pinned(url) {
    const parts = BRANCH_URL.exec(url);
    if (parts === null) {
        return url;
    }
    const repo = new URL(url).pathname.split("/").slice(1, 3).join("/");
    const revision = revisions.get(repo);
    return revision === undefined ? url : `${parts[1]}${revision}${parts[2]}`;
}

// the page installs every model file into the cache, so a miss is a 404, never a download
env.fetch = async (input) => {
    const url = typeof input === "string" ? input : (input.url ?? String(input));
    const cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(pinned(url), { ignoreMethod: true });
    return hit ?? new Response(null, { status: 404 });
};
