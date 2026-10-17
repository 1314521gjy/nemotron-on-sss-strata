// src/poly/threads.hpp - where strata-poly's threads run: Windows' SetThreadAffinityMask and Linux's
// pthread_setaffinity_np, as a header only, so the CMake target list stays as it is.
//
// Two switches use it (main.cpp, Options::host_core and Options::aux_cpus):
//   --host-core off|first|last|<n>  off (the default) pins nothing: the engine runs the threads where the OS put
//                               them, exactly as it did before this header existed.  first pins the host thread
//                               (the one that calls Engine::load()) to the first physical core, last to the last
//                               one, <n> to the n'th (0-based); the CPU expert workers then take the logical
//                               processors the host and the aux threads do not, one each, and the host is pinned
//                               before any of them is created.
//   --aux-cpus off|auto|<list>  where the threads that are neither the host nor a CPU expert worker run - the
//                               adaptive tier's follower and the page mapper.  off (the default) leaves them
//                               wherever they were created; auto puts them on the host core's SMT siblings, then
//                               on the SMT siblings of the physical cores from the last one down, then on the
//                               cores nearest the host core's end; <list> is a list of logical processors (as the
//                               OS numbers them), "15,13" or "12-15".  It pins threads even with --host-core off.
//
// Pinning changes where a thread runs, never what it computes: no arithmetic, no kernel argument and no barrier
// order depends on it.  The switches exist because Windows sends a GPU's interrupts to one logical processor,
// usually the first, and every copy that lands raises one: a host thread spinning there waits for them.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#else
#error "strata-poly's thread placement is written for Windows and Linux"
#endif

namespace strata::poly::affinity {

/// The logical processors of the first processor group: the ones a process that is not group aware runs on.
inline int logical_cpus() {
#if defined(_WIN32)
    const DWORD n = GetActiveProcessorCount(0);
    return n ? (int) n : 1;
#elif defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) return std::max(CPU_COUNT(&set), 1);
    return std::max((int) sysconf(_SC_NPROCESSORS_ONLN), 1);
#endif
}

/// The logical processors of each physical core, in order: an SMT core lists all of them, its first one being
/// the processor this file pins a thread of that core to.  An OS that will not say - and Linux, which this build
/// does not ask - gives one logical processor per core.
inline std::vector<std::vector<int>> physical_cores() {
    const int n = logical_cpus();
    std::vector<std::vector<int>> cores;
#if defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);   // fills `bytes`, returns false
    if (bytes > 0) {
        std::vector<uint8_t> buf((size_t) bytes);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &bytes)) {
            for (size_t at = 0; at < (size_t) bytes;) {
                const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + at);
                if (e->Relationship == RelationProcessorCore) {
                    std::vector<int> cpus;
                    for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
                        const KAFFINITY mask = e->Processor.GroupMask[g].Mask;
                        const int base = (int) e->Processor.GroupMask[g].Group * 64;
                        for (int b = 0; b < 64; ++b)
                            if (mask & ((KAFFINITY) 1 << b)) cpus.push_back(base + b);
                    }
                    if (!cpus.empty()) cores.push_back(std::move(cpus));
                }
                if (e->Size == 0) break;   // never: a record of no size would not advance `at`
                at += e->Size;
            }
        }
    }
#endif
    if (cores.empty())   // an OS that did not say, or one core a logical processor
        for (int c = 0; c < n; ++c) cores.push_back({c});
    return cores;
}

/// Pins the calling thread to one logical processor.  False when the OS refused it (or this build has no
/// placement): the thread then runs wherever it did.
inline bool pin_this_thread(int cpu) {
    if (cpu < 0) return false;
#if defined(_WIN32)
    if (cpu >= 64) return false;   // one processor group is 64 logical processors
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu) != 0;
#elif defined(__linux__)
    if (cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

/// What the switches asked for, worked out on this machine before any of the threads that read it is created.
struct Plan {
    int host = -1;               ///< the host thread's logical processor
    std::vector<int> workers;    ///< the CPU expert workers' logical processors, by `workers % size()` per index
    std::vector<int> aux;        ///< the auxiliary threads' logical processors, by `index % size()` per arrival
    bool aux_on = false;         ///< --aux-cpus asked for placement
};

/// The one plan of the process: Engine::load() makes it before the first thread that reads it.
inline Plan& plan() {
    static Plan p;
    return p;
}

/// The logical processor the i'th CPU expert worker is pinned to; -1 when the plan has none (the tests, or a
/// machine that would not say).
inline int worker_cpu(int i) {
    const Plan& p = plan();
    if (i < 0 || p.workers.empty()) return -1;
    return p.workers[(size_t) i % p.workers.size()];
}

/// Pins the calling worker thread; counted in creation order across every Workers pool of the process.  A pool
/// made after the plan's processors are taken shares them round robin.
inline bool pin_worker_thread() {
    static std::atomic<int> next{0};
    return pin_this_thread(worker_cpu(next.fetch_add(1)));
}

/// Pins the calling auxiliary thread; counted in arrival order.
inline bool pin_aux_thread() {
    Plan& p = plan();
    if (!p.aux_on || p.aux.empty()) return false;
    static std::atomic<int> next{0};
    return pin_this_thread(p.aux[(size_t) next.fetch_add(1) % p.aux.size()]);
}

/// A list of logical processors: "15,13", "12-15", "0,2-3".  Throws std::runtime_error when it is not one.
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    size_t at = 0;
    while (at < s.size()) {
        while (at < s.size() && (s[at] == ' ' || s[at] == ',')) ++at;
        if (at >= s.size()) break;
        char* end = nullptr;
        const long a = std::strtol(s.c_str() + at, &end, 10);
        if (end == s.c_str() + at || a < 0) throw std::runtime_error("--aux-cpus: not a CPU list: " + s);
        at = (size_t) (end - s.c_str());
        long b = a;
        if (at < s.size() && s[at] == '-') {
            const char* from = s.c_str() + at + 1;
            b = std::strtol(from, &end, 10);
            if (end == from || b < 0) throw std::runtime_error("--aux-cpus: not a CPU list: " + s);
            at = (size_t) (end - s.c_str());
        }
        if (b < a) throw std::runtime_error("--aux-cpus: a range that ends before it starts: " + s);
        for (long c = a; c <= b; ++c) out.push_back((int) c);
    }
    return out;
}

/// Works the two switches out against this machine, stores the plan and returns one line that says who goes
/// where.  `workers` is how many CPU expert worker threads will be made.  Throws std::runtime_error for a core
/// or a CPU this machine does not have.
inline std::string plan_threads(const std::string& host_core, const std::string& aux_cpus, int workers) {
    const std::vector<std::vector<int>> cores = physical_cores();
    const int n = logical_cpus();
    Plan p;

    // --host-core: off (the default) pins nothing; first, last or <n> pins the host thread to that physical core
    // and the CPU expert workers to the logical processors the host and the aux threads do not take.
    const bool place = !(host_core.empty() || host_core == "off");
    int core = 0;
    if (host_core.empty() || host_core == "off" || host_core == "first") {
        core = 0;   // a reference core for the aux threads even when the host is not pinned
    } else if (host_core == "last") {
        core = (int) cores.size() - 1;
    } else {
        char* end = nullptr;
        const long c = std::strtol(host_core.c_str(), &end, 10);
        if (end == host_core.c_str() || *end != '\0' || c < 0 || c >= (long) cores.size())
            throw std::runtime_error("--host-core takes off, first, last or a physical core 0 to " +
                                     std::to_string(cores.size() - 1) + ", not " + host_core);
        core = (int) c;
    }
    if (place) p.host = cores[(size_t) core].front();

    // --aux-cpus
    std::vector<int> taken;
    if (place) taken.push_back(p.host);
    if (aux_cpus.empty() || aux_cpus == "off") {
        p.aux_on = false;
    } else if (aux_cpus == "auto") {
        p.aux_on = true;
        // the host core's SMT siblings first, then the SMT siblings of the physical cores from the last one down
        for (size_t s = 1; s < cores[(size_t) core].size() && p.aux.size() < 2; ++s)
            p.aux.push_back(cores[(size_t) core][s]);
        for (int c = (int) cores.size() - 1; c >= 0 && p.aux.size() < 2; --c) {
            if (c == core) continue;
            for (size_t s = 1; s < cores[(size_t) c].size() && p.aux.size() < 2; ++s)
                p.aux.push_back(cores[(size_t) c][s]);
        }
        // a machine with no SMT sibling to give: whole cores from the last one down (the host core's end)
        for (int c = (int) cores.size() - 1; c >= 0 && p.aux.size() < 2; --c)
            if (c != core && std::find(p.aux.begin(), p.aux.end(), cores[(size_t) c].front()) == p.aux.end())
                p.aux.push_back(cores[(size_t) c].front());
        taken.insert(taken.end(), p.aux.begin(), p.aux.end());
    } else {
        p.aux_on = true;
        p.aux = parse_cpu_list(aux_cpus);
        if (p.aux.empty()) throw std::runtime_error("--aux-cpus takes off, auto or a CPU list, not " + aux_cpus);
        for (int c : p.aux)
            if (c < 0 || c >= n)
                throw std::runtime_error("--aux-cpus names CPU " + std::to_string(c) + "; this PC has 0 to " +
                                         std::to_string(n - 1));
        taken.insert(taken.end(), p.aux.begin(), p.aux.end());
    }

    // the CPU expert workers: every logical processor that neither the host nor an aux thread took, in order
    if (place)
        for (int c = 0; c < n; ++c)
            if (std::find(taken.begin(), taken.end(), c) == taken.end()) p.workers.push_back(c);

    plan() = p;

    const std::string host_name = host_core.empty() ? std::string("off") : host_core;
    std::string s;
    if (!place) {
        s = "--host-core " + host_name + " --aux-cpus " + (aux_cpus.empty() ? std::string("off") : aux_cpus) +
            ": host and workers not pinned, aux ";
    } else {
        s = "--host-core " + host_name + " --aux-cpus " + (aux_cpus.empty() ? std::string("off") : aux_cpus) +
            ": host cpu " + std::to_string(p.host) + " (physical core " + std::to_string(core) + " of " +
            std::to_string(cores.size()) + "), " + std::to_string(workers) + " workers on " +
            std::to_string(p.workers.size()) + " cpus";
        if (workers > (int) p.workers.size() && !p.workers.empty())
            s += " (" + std::to_string(workers - (int) p.workers.size()) + " share)";
        s += ", aux ";
    }
    if (!p.aux_on) {
        s += "off";
    } else {
        s += std::to_string(p.aux.size()) + " cpus:";
        for (size_t i = 0; i < p.aux.size() && i < 8; ++i) s += " " + std::to_string(p.aux[i]);
    }
    return s;
}

}  // namespace strata::poly::affinity
