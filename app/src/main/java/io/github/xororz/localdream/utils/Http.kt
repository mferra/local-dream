package io.github.xororz.localdream.utils

import java.io.IOException
import java.net.Inet6Address
import java.net.InetAddress
import java.net.UnknownHostException
import okhttp3.Dns
import okhttp3.Interceptor
import okhttp3.OkHttpClient

// Single OkHttp instance for the whole app. Per-use-case clients must be
// derived via newBuilder() so they share this instance's dispatcher and
// connection pool instead of each spawning their own thread pools. They also
// inherit the outbound allowlist below, which is the point: every HTTP request
// the app makes goes through it.
object Http {
    val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            // Hostnames: refused before any DNS query or connection, and this
            // runs again for every redirect hop.
            .dns(OutboundAllowlist.dns)
            // IP literals skip Dns, so check the first request before OkHttp
            // connects, and every hop (redirects included) before it sends.
            .addInterceptor(OutboundAllowlist.interceptor)
            .addNetworkInterceptor(OutboundAllowlist.interceptor)
            .build()
    }
}

/**
 * The only places the app may reach over HTTP: the device itself (the native
 * backend on 127.0.0.1), the local network (remote host mode, IP addresses
 * only) and the model download sources. Anything else fails with an
 * IOException instead of leaving the device.
 *
 * A custom download base URL set in the app is blocked unless its domain is
 * added to [ALLOWED_DOMAINS].
 */
object OutboundAllowlist {
    // Each entry also admits its subdomains. Hugging Face redirects file
    // downloads to CDN hosts under these (cdn-lfs.huggingface.co,
    // cas-bridge.xethub.hf.co, ...).
    private val ALLOWED_DOMAINS = listOf(
        "huggingface.co",
        "hf.co",
        "hf-mirror.com",
    )

    // Octets capped at 255: anything else is a hostname to InetAddress, which
    // would resolve it through DNS.
    private const val OCTET = "(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)"
    private val IPV4_LITERAL = Regex("^($OCTET\\.){3}$OCTET$")

    fun isAllowed(host: String): Boolean {
        val h = host.lowercase().trimEnd('.')
        if (h == "localhost") return true
        if (isIpLiteral(h)) return isLocalAddress(InetAddress.getByName(h))
        return ALLOWED_DOMAINS.any { h == it || h.endsWith(".$it") }
    }

    val dns = object : Dns {
        override fun lookup(hostname: String): List<InetAddress> {
            if (!isAllowed(hostname)) {
                throw UnknownHostException("Blocked by outbound allowlist: $hostname")
            }
            return Dns.SYSTEM.lookup(hostname)
        }
    }

    val interceptor = Interceptor { chain ->
        val host = chain.request().url.host
        if (!isAllowed(host)) throw IOException("Blocked by outbound allowlist: $host")
        chain.proceed(chain.request())
    }

    // OkHttp hands IPv6 hosts over without brackets, so any ':' means a literal.
    // Parsing a literal never triggers a DNS query.
    private fun isIpLiteral(host: String): Boolean = IPV4_LITERAL.matches(host) || ':' in host

    private fun isLocalAddress(address: InetAddress): Boolean = address.isLoopbackAddress ||
        address.isSiteLocalAddress ||
        address.isLinkLocalAddress ||
        // IPv6 unique local addresses (fc00::/7), the v6 counterpart of the
        // private ranges above; isSiteLocalAddress only knows deprecated fec0::/10.
        (address is Inet6Address && (address.address[0].toInt() and 0xFE) == 0xFC)
}
