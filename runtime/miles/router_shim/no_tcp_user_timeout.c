// LD_PRELOAD shim for the Rust sgl-router: ignore setsockopt(IPPROTO_TCP, TCP_USER_TIMEOUT) so the
// kernel default (0 = off) stays. sgl-router 0.3.2 builds its reqwest 0.12.28 client with
// TCP_USER_TIMEOUT=30s; together with TCP keepalive, one lost keepalive probe then aborts a healthy
// in-flight /generate. Same effect as .tcp_user_timeout(None) in the router's client builder.
// Preloaded into the router process only, by oi_router_preload.py; see docs/miles/operations.md.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <sys/socket.h>

int setsockopt(int fd, int level, int optname, const void *optval, socklen_t optlen) {
    static int (*real)(int, int, int, const void *, socklen_t) = 0;
    if (!real) real = (int (*)(int, int, int, const void *, socklen_t))dlsym(RTLD_NEXT, "setsockopt");
    if (level == IPPROTO_TCP && optname == TCP_USER_TIMEOUT) {
        static int logged = 0;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED) && optval && optlen >= sizeof(unsigned)) {
            fprintf(stderr, "[no_tcp_user_timeout] dropped TCP_USER_TIMEOUT=%u ms\n", *(const unsigned *)optval);
        }
        return 0;
    }
    if (!real) {
        errno = ENOSYS;
        return -1;
    }
    return real(fd, level, optname, optval, optlen);
}
