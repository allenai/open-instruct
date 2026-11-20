// LD_PRELOAD shim: ignore setsockopt(IPPROTO_TCP, TCP_USER_TIMEOUT) so the kernel default (0 = off) stays.
// Emulates patching sgl-router's reqwest client with .tcp_user_timeout(None).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <stdio.h>

int setsockopt(int fd, int level, int optname, const void *optval, socklen_t optlen) {
    static int (*real)(int, int, int, const void *, socklen_t) = 0;
    if (!real) real = dlsym(RTLD_NEXT, "setsockopt");
    if (level == IPPROTO_TCP && optname == TCP_USER_TIMEOUT) {
        static int said = 0;
        if (!said++) fprintf(stderr, "[shim] dropped TCP_USER_TIMEOUT=%u ms\n", *(const unsigned *)optval);
        return 0;
    }
    return real(fd, level, optname, optval, optlen);
}
