# NetworkingLib API Specification

The NetworkingLib provides comprehensive networking capabilities for STARK applications, including HTTP clients and servers, WebSocket support, gRPC services, and low-level networking primitives. It's designed for high-performance, async-first networking with built-in security and observability.

## HTTP Client and Server

### HTTP Client

```stark
module net::http {
    // HTTP client for making requests
    struct Client {
        timeout: Duration,
        max_redirects: i32,
        user_agent: str,
        default_headers: Map<str, str>,
        cookies: CookieJar,
        proxy: ?Proxy,
        tls_config: ?TlsConfig
    }
    
    impl Client {
        fn new() -> Self
        fn with_timeout(timeout: Duration) -> Self
        fn with_user_agent(user_agent: str) -> Self
        fn with_proxy(proxy: Proxy) -> Self
        fn with_tls_config(config: TlsConfig) -> Self
        
        // Basic HTTP methods
        async fn get(url: str) -> Result<Response, HttpError>
        async fn post(url: str, body: Body) -> Result<Response, HttpError>
        async fn put(url: str, body: Body) -> Result<Response, HttpError>
        async fn delete(url: str) -> Result<Response, HttpError>
        async fn head(url: str) -> Result<Response, HttpError>
        async fn patch(url: str, body: Body) -> Result<Response, HttpError>
        async fn options(url: str) -> Result<Response, HttpError>
        
        // Advanced request building
        fn request(method: Method, url: str) -> RequestBuilder
        async fn send(request: Request) -> Result<Response, HttpError>
        
        // Streaming requests
        async fn stream_get(url: str) -> Result<ResponseStream, HttpError>
        async fn upload_stream(url: str, stream: InputStream) -> Result<Response, HttpError>
        
        // Batch requests
        async fn send_batch(requests: [Request]) -> [Result<Response, HttpError>]
    }
    
    // HTTP request builder
    struct RequestBuilder {
        method: Method,
        url: Url,
        headers: Map<str, str>,
        body: ?Body,
        timeout: ?Duration,
        query_params: Map<str, str>
    }
    
    impl RequestBuilder {
        fn header(key: str, value: str) -> Self
        fn headers(headers: Map<str, str>) -> Self
        fn query(key: str, value: str) -> Self
        fn query_params(params: Map<str, str>) -> Self
        fn body(body: Body) -> Self
        fn json<T: Serialize>(data: T) -> Self
        fn form(data: Map<str, str>) -> Self
        fn multipart(form: MultipartForm) -> Self
        fn timeout(timeout: Duration) -> Self
        fn bearer_auth(token: str) -> Self
        fn basic_auth(username: str, password: str) -> Self
        
        async fn send() -> Result<Response, HttpError>
    }
    
    // HTTP request
    struct Request {
        method: Method,
        url: Url,
        headers: Headers,
        body: ?Body,
        version: HttpVersion
    }
    
    // HTTP response
    struct Response {
        status: StatusCode,
        headers: Headers,
        body: Body,
        version: HttpVersion,
        url: Url
    }
    
    impl Response {
        fn status() -> StatusCode
        fn headers() -> &Headers
        fn content_length() -> ?i64
        
        // Body consumption methods
        async fn text() -> Result<str, HttpError>
        async fn bytes() -> Result<[u8], HttpError>
        async fn json<T: Deserialize>() -> Result<T, HttpError>
        async fn stream() -> Result<BodyStream, HttpError>
        
        // Response utilities
        fn is_success() -> bool
        fn is_redirect() -> bool
        fn is_client_error() -> bool
        fn is_server_error() -> bool
    }
    
    // HTTP methods
    enum Method {
        GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH, TRACE, CONNECT
    }
    
    // Status codes
    enum StatusCode {
        Continue = 100,
        Ok = 200,
        Created = 201,
        Accepted = 202,
        NoContent = 204,
        MovedPermanently = 301,
        Found = 302,
        NotModified = 304,
        BadRequest = 400,
        Unauthorized = 401,
        Forbidden = 403,
        NotFound = 404,
        MethodNotAllowed = 405,
        InternalServerError = 500,
        NotImplemented = 501,
        BadGateway = 502,
        ServiceUnavailable = 503
    }
    
    // HTTP body types
    enum Body {
        Empty,
        Text(str),
        Bytes([u8]),
        Json(Any),
        Form(Map<str, str>),
        Multipart(MultipartForm),
        Stream(InputStream)
    }
}
```

### HTTP Server

```stark
module net::http::server {
    // HTTP server
    struct Server {
        addr: SocketAddr,
        routes: Router,
        middleware: [Middleware],
        tls_config: ?TlsConfig,
        max_connections: i32,
        read_timeout: Duration,
        write_timeout: Duration,
        keep_alive: bool
    }
    
    impl Server {
        fn new(addr: SocketAddr) -> Self
        fn with_tls(config: TlsConfig) -> Self
        fn with_middleware(middleware: Middleware) -> Self
        fn with_routes(router: Router) -> Self
        fn with_max_connections(max: i32) -> Self
        fn with_timeouts(read: Duration, write: Duration) -> Self
        
        async fn serve() -> Result<(), ServerError>
        async fn serve_with_graceful_shutdown(shutdown: impl Future<()>) -> Result<(), ServerError>
        fn shutdown() -> Result<(), ServerError>
    }
    
    // HTTP router
    struct Router {
        routes: [Route],
        not_found_handler: ?Handler,
        error_handler: ?ErrorHandler
    }
    
    impl Router {
        fn new() -> Self
        
        // Route registration
        fn get(path: str, handler: Handler) -> Self
        fn post(path: str, handler: Handler) -> Self
        fn put(path: str, handler: Handler) -> Self
        fn delete(path: str, handler: Handler) -> Self
        fn patch(path: str, handler: Handler) -> Self
        fn options(path: str, handler: Handler) -> Self
        fn head(path: str, handler: Handler) -> Self
        
        // Generic route registration
        fn route(method: Method, path: str, handler: Handler) -> Self
        fn any(path: str, handler: Handler) -> Self
        
        // Route groups and nesting
        fn group(prefix: str) -> RouterGroup
        fn mount(prefix: str, router: Router) -> Self
        
        // Middleware
        fn middleware(middleware: Middleware) -> Self
        
        // Error handling
        fn not_found(handler: Handler) -> Self
        fn error_handler(handler: ErrorHandler) -> Self
    }
    
    // Route handler function type
    type Handler = fn(Request, &mut Response) -> Result<(), HandlerError>
    type AsyncHandler = async fn(Request, &mut Response) -> Result<(), HandlerError>
    type ErrorHandler = fn(Error, Request, &mut Response) -> Result<(), HandlerError>
    
    // Request context with path parameters and state
    struct RequestContext {
        request: Request,
        params: Map<str, str>,
        query: Map<str, str>,
        state: Map<str, Any>,
        remote_addr: SocketAddr
    }
    
    impl RequestContext {
        fn param(key: str) -> ?str
        fn query_param(key: str) -> ?str
        fn header(key: str) -> ?str
        fn get_state<T>(key: str) -> ?T
        fn set_state<T>(key: str, value: T)
        
        // Request body parsing
        async fn json<T: Deserialize>() -> Result<T, ParseError>
        async fn form() -> Result<Map<str, str>, ParseError>
        async fn multipart() -> Result<MultipartForm, ParseError>
        async fn text() -> Result<str, ParseError>
        async fn bytes() -> Result<[u8], ParseError>
    }
    
    // Response builder
    struct ResponseBuilder {
        status: StatusCode,
        headers: Map<str, str>,
        body: ?Body
    }
    
    impl ResponseBuilder {
        fn status(status: StatusCode) -> Self
        fn header(key: str, value: str) -> Self
        fn headers(headers: Map<str, str>) -> Self
        fn body(body: Body) -> Self
        fn json<T: Serialize>(data: T) -> Self
        fn text(text: str) -> Self
        fn html(html: str) -> Self
        fn file(path: str) -> Self
        fn redirect(url: str, permanent: bool = false) -> Self
    }
    
    // Middleware trait
    trait Middleware {
        async fn handle(req: Request, next: NextHandler) -> Result<Response, MiddlewareError>
    }
    
    type NextHandler = async fn(Request) -> Result<Response, HandlerError>
}
```

### Built-in Middleware

```stark
module net::http::middleware {
    // CORS middleware
    struct Cors {
        allowed_origins: [str],
        allowed_methods: [Method],
        allowed_headers: [str],
        exposed_headers: [str],
        max_age: ?Duration,
        allow_credentials: bool
    }
    
    impl Cors {
        fn new() -> Self
        fn allow_origin(origin: str) -> Self
        fn allow_any_origin() -> Self
        fn allow_methods(methods: [Method]) -> Self
        fn allow_headers(headers: [str]) -> Self
        fn expose_headers(headers: [str]) -> Self
        fn max_age(duration: Duration) -> Self
        fn allow_credentials(allow: bool) -> Self
    }
    
    // Rate limiting middleware
    struct RateLimit {
        max_requests: i32,
        window: Duration,
        store: RateLimitStore,
        key_extractor: fn(Request) -> str
    }
    
    impl RateLimit {
        fn new(max_requests: i32, window: Duration) -> Self
        fn with_key_extractor(extractor: fn(Request) -> str) -> Self
        fn with_store(store: RateLimitStore) -> Self
    }
    
    trait RateLimitStore {
        async fn get(key: str) -> ?RateLimitInfo
        async fn set(key: str, info: RateLimitInfo)
    }
    
    // Authentication middleware
    struct BearerAuth {
        verify_token: fn(str) -> Result<Claims, AuthError>
    }
    
    struct BasicAuth {
        verify_credentials: fn(str, str) -> Result<User, AuthError>
    }
    
    struct JwtAuth {
        secret: str,
        algorithm: JwtAlgorithm,
        leeway: Duration
    }
    
    // Logging middleware
    struct Logger {
        format: LogFormat,
        exclude_paths: [str],
        include_request_body: bool,
        include_response_body: bool
    }
    
    enum LogFormat {
        Common, Combined, Custom(str)
    }
    
    // Compression middleware
    struct Compression {
        algorithms: [CompressionAlgorithm],
        min_size: i32,
        exclude_content_types: [str]
    }
    
    enum CompressionAlgorithm {
        Gzip, Deflate, Brotli
    }
    
    // Security headers middleware
    struct SecurityHeaders {
        content_security_policy: ?str,
        strict_transport_security: ?str,
        x_frame_options: ?str,
        x_content_type_options: bool,
        referrer_policy: ?str
    }
    
    // Request ID middleware
    struct RequestId {
        header_name: str,
        generator: fn() -> str
    }
    
    // Timeout middleware
    struct Timeout {
        duration: Duration
    }
}
```

## WebSocket Support

```stark
module net::websocket {
    // WebSocket client
    struct WebSocketClient {
        url: str,
        headers: Map<str, str>,
        protocols: [str],
        timeout: Duration,
        max_message_size: i32,
        max_frame_size: i32
    }
    
    impl WebSocketClient {
        fn new(url: str) -> Self
        fn with_headers(headers: Map<str, str>) -> Self
        fn with_protocols(protocols: [str]) -> Self
        fn with_timeout(timeout: Duration) -> Self
        
        async fn connect() -> Result<WebSocket, WebSocketError>
    }
    
    // WebSocket connection
    struct WebSocket {
        stream: WebSocketStream,
        config: WebSocketConfig
    }
    
    impl WebSocket {
        // Send messages
        async fn send_text(text: str) -> Result<(), WebSocketError>
        async fn send_binary(data: [u8]) -> Result<(), WebSocketError>
        async fn send_ping(data: ?[u8] = null) -> Result<(), WebSocketError>
        async fn send_pong(data: ?[u8] = null) -> Result<(), WebSocketError>
        async fn send_close(code: ?CloseCode = null, reason: ?str = null) -> Result<(), WebSocketError>
        
        // Receive messages
        async fn receive() -> Result<Message, WebSocketError>
        async fn receive_text() -> Result<str, WebSocketError>
        async fn receive_binary() -> Result<[u8], WebSocketError>
        
        // Stream interface
        fn message_stream() -> MessageStream
        fn text_stream() -> TextStream
        fn binary_stream() -> BinaryStream
        
        // Connection management
        async fn close(code: ?CloseCode = null, reason: ?str = null) -> Result<(), WebSocketError>
        fn is_open() -> bool
        fn local_addr() -> SocketAddr
        fn remote_addr() -> SocketAddr
    }
    
    // WebSocket messages
    enum Message {
        Text(str),
        Binary([u8]),
        Ping([u8]),
        Pong([u8]),
        Close(CloseFrame)
    }
    
    struct CloseFrame {
        code: CloseCode,
        reason: str
    }
    
    enum CloseCode {
        Normal = 1000,
        GoingAway = 1001,
        ProtocolError = 1002,
        UnsupportedData = 1003,
        InvalidFramePayloadData = 1007,
        PolicyViolation = 1008,
        MessageTooBig = 1009,
        MandatoryExtension = 1010,
        InternalServerError = 1011
    }
    
    // WebSocket server
    struct WebSocketServer {
        router: WebSocketRouter,
        config: WebSocketServerConfig
    }
    
    impl WebSocketServer {
        fn new() -> Self
        fn route(path: str, handler: WebSocketHandler) -> Self
        fn middleware(middleware: WebSocketMiddleware) -> Self
        
        async fn serve(addr: SocketAddr) -> Result<(), ServerError>
    }
    
    type WebSocketHandler = async fn(WebSocket, Request) -> Result<(), WebSocketError>
    
    // WebSocket upgrade for HTTP servers
    fn upgrade_websocket(
        request: Request,
        handler: WebSocketHandler
    ) -> Result<Response, WebSocketError>
}
```

## gRPC Support

```stark
module net::grpc {
    // gRPC client
    struct Client {
        endpoint: str,
        tls_config: ?TlsConfig,
        timeout: Duration,
        max_message_size: i32,
        compression: ?CompressionType,
        metadata: Metadata,
        interceptors: [ClientInterceptor]
    }
    
    impl Client {
        fn new(endpoint: str) -> Self
        fn with_tls(config: TlsConfig) -> Self
        fn with_timeout(timeout: Duration) -> Self
        fn with_compression(compression: CompressionType) -> Self
        fn with_metadata(metadata: Metadata) -> Self
        fn with_interceptor(interceptor: ClientInterceptor) -> Self
        
        // Unary call
        async fn call<Req, Resp>(
            method: str,
            request: Req
        ) -> Result<Resp, GrpcError>
        where Req: Serialize, Resp: Deserialize
        
        // Server streaming
        async fn server_streaming<Req, Resp>(
            method: str,
            request: Req
        ) -> Result<ResponseStream<Resp>, GrpcError>
        where Req: Serialize, Resp: Deserialize
        
        // Client streaming
        async fn client_streaming<Req, Resp>(
            method: str,
            requests: RequestStream<Req>
        ) -> Result<Resp, GrpcError>
        where Req: Serialize, Resp: Deserialize
        
        // Bidirectional streaming
        async fn bidirectional_streaming<Req, Resp>(
            method: str,
            requests: RequestStream<Req>
        ) -> Result<ResponseStream<Resp>, GrpcError>
        where Req: Serialize, Resp: Deserialize
    }
    
    // gRPC server
    struct Server {
        addr: SocketAddr,
        services: [Service],
        interceptors: [ServerInterceptor],
        tls_config: ?TlsConfig,
        max_connections: i32,
        max_message_size: i32,
        compression: [CompressionType]
    }
    
    impl Server {
        fn new(addr: SocketAddr) -> Self
        fn add_service(service: Service) -> Self
        fn add_interceptor(interceptor: ServerInterceptor) -> Self
        fn with_tls(config: TlsConfig) -> Self
        fn with_max_connections(max: i32) -> Self
        fn with_compression(types: [CompressionType]) -> Self
        
        async fn serve() -> Result<(), ServerError>
        async fn serve_with_shutdown(shutdown: impl Future<()>) -> Result<(), ServerError>
    }
    
    // Service definition
    trait Service {
        fn name() -> str
        fn methods() -> [ServiceMethod]
        async fn call(method: str, request: Request) -> Result<Response, ServiceError>
    }
    
    struct ServiceMethod {
        name: str,
        input_type: Type,
        output_type: Type,
        client_streaming: bool,
        server_streaming: bool
    }
    
    // Request and response types
    struct Request<T> {
        message: T,
        metadata: Metadata,
        peer: SocketAddr
    }
    
    struct Response<T> {
        message: T,
        metadata: Metadata,
        status: Status
    }
    
    // gRPC status
    struct Status {
        code: StatusCode,
        message: str,
        details: [Any]
    }
    
    enum StatusCode {
        Ok = 0,
        Cancelled = 1,
        Unknown = 2,
        InvalidArgument = 3,
        DeadlineExceeded = 4,
        NotFound = 5,
        AlreadyExists = 6,
        PermissionDenied = 7,
        ResourceExhausted = 8,
        FailedPrecondition = 9,
        Aborted = 10,
        OutOfRange = 11,
        Unimplemented = 12,
        Internal = 13,
        Unavailable = 14,
        DataLoss = 15,
        Unauthenticated = 16
    }
    
    // Metadata (headers)
    struct Metadata {
        entries: Map<str, [str]>
    }
    
    impl Metadata {
        fn new() -> Self
        fn insert(key: str, value: str) -> Self
        fn append(key: str, value: str) -> Self
        fn get(key: str) -> ?[str]
        fn remove(key: str) -> ?[str]
    }
    
    // Compression types
    enum CompressionType {
        Gzip, Deflate, None
    }
    
    // Interceptors
    trait ClientInterceptor {
        async fn intercept<Req, Resp>(
            request: Request<Req>,
            next: NextClientCall<Req, Resp>
        ) -> Result<Response<Resp>, GrpcError>
    }
    
    trait ServerInterceptor {
        async fn intercept<Req, Resp>(
            request: Request<Req>,
            next: NextServerCall<Req, Resp>
        ) -> Result<Response<Resp>, ServiceError>
    }
    
    // Streaming types
    trait RequestStream<T> {
        async fn next() -> ?Result<T, StreamError>
    }
    
    trait ResponseStream<T> {
        async fn next() -> ?Result<T, StreamError>
    }
}
```

## TCP/UDP Sockets

```stark
module net::socket {
    // TCP listener
    struct TcpListener {
        addr: SocketAddr,
        backlog: i32,
        reuse_addr: bool,
        reuse_port: bool,
        nodelay: bool
    }
    
    impl TcpListener {
        async fn bind(addr: SocketAddr) -> Result<Self, SocketError>
        fn with_backlog(backlog: i32) -> Self
        fn with_reuse_addr(reuse: bool) -> Self
        fn with_reuse_port(reuse: bool) -> Self
        fn with_nodelay(nodelay: bool) -> Self
        
        async fn accept() -> Result<(TcpStream, SocketAddr), SocketError>
        fn local_addr() -> SocketAddr
        fn set_ttl(ttl: u32) -> Result<(), SocketError>
    }
    
    // TCP stream
    struct TcpStream {
        peer_addr: SocketAddr,
        local_addr: SocketAddr,
        nodelay: bool,
        keepalive: ?Duration,
        linger: ?Duration
    }
    
    impl TcpStream {
        async fn connect(addr: SocketAddr) -> Result<Self, SocketError>
        async fn connect_timeout(addr: SocketAddr, timeout: Duration) -> Result<Self, SocketError>
        
        // Reading
        async fn read(buf: &mut [u8]) -> Result<i32, SocketError>
        async fn read_exact(buf: &mut [u8]) -> Result<(), SocketError>
        async fn read_to_end() -> Result<[u8], SocketError>
        async fn read_to_string() -> Result<str, SocketError>
        
        // Writing
        async fn write(buf: [u8]) -> Result<i32, SocketError>
        async fn write_all(buf: [u8]) -> Result<(), SocketError>
        async fn flush() -> Result<(), SocketError>
        
        // Stream interface
        fn reader() -> TcpReader
        fn writer() -> TcpWriter
        
        // Configuration
        fn set_nodelay(nodelay: bool) -> Result<(), SocketError>
        fn set_keepalive(keepalive: ?Duration) -> Result<(), SocketError>
        fn set_linger(linger: ?Duration) -> Result<(), SocketError>
        fn set_ttl(ttl: u32) -> Result<(), SocketError>
        
        // Connection info
        fn peer_addr() -> SocketAddr
        fn local_addr() -> SocketAddr
        fn is_connected() -> bool
        
        // Shutdown
        async fn shutdown() -> Result<(), SocketError>
    }
    
    // UDP socket
    struct UdpSocket {
        local_addr: SocketAddr,
        broadcast: bool,
        multicast_loop: bool,
        multicast_ttl: u32
    }
    
    impl UdpSocket {
        async fn bind(addr: SocketAddr) -> Result<Self, SocketError>
        async fn connect(addr: SocketAddr) -> Result<Self, SocketError>
        
        // Sending
        async fn send_to(buf: [u8], addr: SocketAddr) -> Result<i32, SocketError>
        async fn send(buf: [u8]) -> Result<i32, SocketError>  // For connected sockets
        
        // Receiving
        async fn recv_from(buf: &mut [u8]) -> Result<(i32, SocketAddr), SocketError>
        async fn recv(buf: &mut [u8]) -> Result<i32, SocketError>  // For connected sockets
        
        // Multicast
        fn join_multicast_v4(multiaddr: Ipv4Addr, interface: Ipv4Addr) -> Result<(), SocketError>
        fn leave_multicast_v4(multiaddr: Ipv4Addr, interface: Ipv4Addr) -> Result<(), SocketError>
        fn join_multicast_v6(multiaddr: Ipv6Addr, interface: u32) -> Result<(), SocketError>
        fn leave_multicast_v6(multiaddr: Ipv6Addr, interface: u32) -> Result<(), SocketError>
        
        // Configuration
        fn set_broadcast(broadcast: bool) -> Result<(), SocketError>
        fn set_multicast_loop_v4(loop: bool) -> Result<(), SocketError>
        fn set_multicast_ttl_v4(ttl: u32) -> Result<(), SocketError>
        fn set_ttl(ttl: u32) -> Result<(), SocketError>
        
        fn local_addr() -> SocketAddr
    }
    
    // Socket addresses
    enum SocketAddr {
        V4(SocketAddrV4),
        V6(SocketAddrV6)
    }
    
    struct SocketAddrV4 {
        ip: Ipv4Addr,
        port: u16
    }
    
    struct SocketAddrV6 {
        ip: Ipv6Addr,
        port: u16,
        flowinfo: u32,
        scope_id: u32
    }
    
    // IP addresses
    struct Ipv4Addr([u8; 4])
    struct Ipv6Addr([u16; 8])
    
    impl Ipv4Addr {
        fn new(a: u8, b: u8, c: u8, d: u8) -> Self
        fn localhost() -> Self  // 127.0.0.1
        fn unspecified() -> Self  // 0.0.0.0
        fn broadcast() -> Self  // 255.255.255.255
        
        fn is_private() -> bool
        fn is_loopback() -> bool
        fn is_multicast() -> bool
        fn is_broadcast() -> bool
    }
    
    impl Ipv6Addr {
        fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Self
        fn localhost() -> Self  // ::1
        fn unspecified() -> Self  // ::
        
        fn is_loopback() -> bool
        fn is_multicast() -> bool
        fn is_unicast() -> bool
    }
}
```

## TLS/SSL Support

```stark
module net::tls {
    // TLS configuration
    struct TlsConfig {
        certificates: [Certificate],
        private_key: PrivateKey,
        ca_certificates: [Certificate],
        client_auth: ClientAuth,
        protocols: [TlsVersion],
        cipher_suites: [CipherSuite],
        alpn_protocols: [str],
        server_name: ?str,
        insecure_skip_verify: bool
    }
    
    impl TlsConfig {
        fn new() -> Self
        fn with_certificate(cert: Certificate, key: PrivateKey) -> Self
        fn with_ca_certificate(ca: Certificate) -> Self
        fn with_client_auth(auth: ClientAuth) -> Self
        fn with_protocols(protocols: [TlsVersion]) -> Self
        fn with_cipher_suites(suites: [CipherSuite]) -> Self
        fn with_alpn_protocols(protocols: [str]) -> Self
        fn with_server_name(name: str) -> Self
        fn insecure_skip_verify(skip: bool) -> Self
        
        // Load from files
        fn load_certificate_from_file(path: str) -> Result<Certificate, TlsError>
        fn load_private_key_from_file(path: str) -> Result<PrivateKey, TlsError>
        fn load_ca_certificates_from_file(path: str) -> Result<[Certificate], TlsError>
    }
    
    // Certificate types
    struct Certificate {
        der: [u8],
        pem: str
    }
    
    struct PrivateKey {
        der: [u8],
        pem: str
    }
    
    enum ClientAuth {
        NoClientCert,
        RequestClientCert,
        RequireAnyClientCert,
        VerifyClientCertIfGiven,
        RequireAndVerifyClientCert
    }
    
    enum TlsVersion {
        TLS1_0, TLS1_1, TLS1_2, TLS1_3
    }
    
    enum CipherSuite {
        TLS_AES_128_GCM_SHA256,
        TLS_AES_256_GCM_SHA384,
        TLS_CHACHA20_POLY1305_SHA256,
        TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
        TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
    }
    
    // TLS streams
    struct TlsStream {
        inner: TcpStream,
        config: TlsConfig,
        peer_certificates: [Certificate],
        negotiated_protocol: ?str,
        cipher_suite: CipherSuite,
        tls_version: TlsVersion
    }
    
    impl TlsStream {
        async fn connect(stream: TcpStream, config: TlsConfig) -> Result<Self, TlsError>
        async fn accept(stream: TcpStream, config: TlsConfig) -> Result<Self, TlsError>
        
        // Same read/write interface as TcpStream
        async fn read(buf: &mut [u8]) -> Result<i32, TlsError>
        async fn write(buf: [u8]) -> Result<i32, TlsError>
        async fn shutdown() -> Result<(), TlsError>
        
        // TLS-specific methods
        fn peer_certificates() -> [Certificate]
        fn negotiated_protocol() -> ?str
        fn cipher_suite() -> CipherSuite
        fn tls_version() -> TlsVersion
    }
}
```

## DNS Resolution

```stark
module net::dns {
    // DNS resolver
    struct Resolver {
        servers: [SocketAddr],
        timeout: Duration,
        attempts: i32,
        use_hosts_file: bool,
        use_system_config: bool
    }
    
    impl Resolver {
        fn new() -> Self
        fn with_servers(servers: [SocketAddr]) -> Self
        fn with_timeout(timeout: Duration) -> Self
        fn with_attempts(attempts: i32) -> Self
        fn use_hosts_file(use_hosts: bool) -> Self
        fn use_system_config(use_system: bool) -> Self
        
        // DNS queries
        async fn lookup_host(name: str) -> Result<[IpAddr], DnsError>
        async fn lookup_addr(addr: IpAddr) -> Result<str, DnsError>
        async fn lookup_mx(name: str) -> Result<[MxRecord], DnsError>
        async fn lookup_txt(name: str) -> Result<[TxtRecord], DnsError>
        async fn lookup_srv(name: str) -> Result<[SrvRecord], DnsError>
        async fn lookup_cname(name: str) -> Result<str, DnsError>
        
        // Generic record lookup
        async fn lookup<T: DnsRecord>(name: str, record_type: RecordType) -> Result<[T], DnsError>
    }
    
    // DNS record types
    trait DnsRecord {
        fn record_type() -> RecordType
        fn from_bytes(data: [u8]) -> Result<Self, DnsError>
    }
    
    enum RecordType {
        A = 1,
        NS = 2,
        CNAME = 5,
        MX = 15,
        TXT = 16,
        AAAA = 28,
        SRV = 33
    }
    
    struct MxRecord {
        preference: u16,
        exchange: str
    }
    
    struct TxtRecord {
        data: str
    }
    
    struct SrvRecord {
        priority: u16,
        weight: u16,
        port: u16,
        target: str
    }
    
    // High-level DNS functions
    async fn resolve_host(name: str) -> Result<[IpAddr], DnsError>
    async fn resolve_addr(addr: IpAddr) -> Result<str, DnsError>
    
    enum IpAddr {
        V4(Ipv4Addr),
        V6(Ipv6Addr)
    }
}
```

## Connection Pooling

```stark
module net::pool {
    // Connection pool for HTTP clients
    struct HttpPool {
        max_idle_per_host: i32,
        max_idle_total: i32,
        idle_timeout: Duration,
        max_lifetime: Duration,
        keep_alive: bool
    }
    
    impl HttpPool {
        fn new() -> Self
        fn with_max_idle_per_host(max: i32) -> Self
        fn with_max_idle_total(max: i32) -> Self
        fn with_idle_timeout(timeout: Duration) -> Self
        fn with_max_lifetime(lifetime: Duration) -> Self
        fn with_keep_alive(keep_alive: bool) -> Self
        
        async fn get_connection(host: str) -> Result<HttpConnection, PoolError>
        fn return_connection(conn: HttpConnection)
        fn close_idle_connections()
        fn stats() -> PoolStats
    }
    
    // Generic connection pool
    struct ConnectionPool<T> {
        factory: ConnectionFactory<T>,
        max_size: i32,
        min_size: i32,
        max_lifetime: Duration,
        idle_timeout: Duration,
        connection_timeout: Duration
    }
    
    impl<T> ConnectionPool<T> {
        fn new(factory: ConnectionFactory<T>) -> Self
        fn with_max_size(max: i32) -> Self
        fn with_min_size(min: i32) -> Self
        fn with_max_lifetime(lifetime: Duration) -> Self
        fn with_idle_timeout(timeout: Duration) -> Self
        fn with_connection_timeout(timeout: Duration) -> Self
        
        async fn get() -> Result<PooledConnection<T>, PoolError>
        fn size() -> i32
        fn idle_count() -> i32
        fn active_count() -> i32
        fn stats() -> PoolStats
        fn health_check() -> PoolHealth
    }
    
    trait ConnectionFactory<T> {
        async fn create() -> Result<T, ConnectionError>
        async fn validate(conn: &T) -> bool
    }
    
    struct PooledConnection<T> {
        inner: T,
        pool: &ConnectionPool<T>,
        created_at: DateTime,
        last_used: DateTime
    }
    
    struct PoolStats {
        total_connections: i32,
        active_connections: i32,
        idle_connections: i32,
        created_connections: i64,
        closed_connections: i64,
        failed_connections: i64
    }
    
    enum PoolHealth {
        Healthy,
        Warning { message: str },
        Critical { message: str }
    }
}
```

## Error Handling

```stark
// Networking errors
enum NetworkError {
    HttpError(HttpError),
    WebSocketError(WebSocketError),
    GrpcError(GrpcError),
    SocketError(SocketError),
    TlsError(TlsError),
    DnsError(DnsError),
    PoolError(PoolError)
}

enum HttpError {
    Timeout,
    ConnectionFailed { reason: str },
    InvalidUrl { url: str },
    InvalidResponse { reason: str },
    TooManyRedirects,
    BodyTooLarge { size: i64, limit: i64 },
    Decode { format: str, reason: str },
    Io { error: IOError }
}

enum WebSocketError {
    ConnectionFailed { reason: str },
    ProtocolError { message: str },
    InvalidMessage { reason: str },
    ConnectionClosed { code: CloseCode, reason: str },
    MessageTooLarge { size: i32, limit: i32 },
    Timeout
}

enum GrpcError {
    Status { status: Status },
    Transport { reason: str },
    Timeout,
    Cancelled,
    InvalidMessage { reason: str },
    ServiceNotFound { service: str },
    MethodNotFound { method: str }
}

enum SocketError {
    AddressInUse { addr: SocketAddr },
    AddressNotAvailable { addr: SocketAddr },
    ConnectionRefused { addr: SocketAddr },
    ConnectionReset,
    ConnectionAborted,
    NotConnected,
    Timeout,
    Io { error: IOError }
}

enum TlsError {
    HandshakeFailed { reason: str },
    CertificateError { reason: str },
    ProtocolError { reason: str },
    InvalidConfiguration { reason: str },
    Io { error: IOError }
}

enum DnsError {
    NotFound { name: str },
    Timeout,
    ServerFailure,
    InvalidName { name: str },
    NoServers,
    Io { error: IOError }
}

enum PoolError {
    Timeout,
    ConnectionFailed { reason: str },
    PoolExhausted,
    ConnectionInvalid,
    PoolClosed
}
```

## Examples

```stark
// HTTP client example
async fn example_http_client() {
    let client = Client::new()
        .with_timeout(Duration::seconds(30))
        .with_user_agent("STARK-App/1.0");
    
    // Simple GET request
    let response = client.get("https://api.example.com/users").await?;
    let users: [User] = response.json().await?;
    
    // POST with JSON
    let new_user = User { name: "Alice", email: "alice@example.com" };
    let response = client
        .request(Method::POST, "https://api.example.com/users")
        .json(new_user)
        .bearer_auth("token123")
        .send()
        .await?;
    
    print(f"Created user: {response.status()}");
}

// HTTP server example
async fn example_http_server() {
    let router = Router::new()
        .get("/", hello_handler)
        .post("/users", create_user_handler)
        .get("/users/:id", get_user_handler)
        .middleware(Logger::new())
        .middleware(Cors::new().allow_any_origin());
    
    let server = Server::new("127.0.0.1:8080".parse()?)
        .with_routes(router)
        .with_max_connections(1000);
    
    print("Starting server on :8080");
    server.serve().await?;
}

async fn hello_handler(req: RequestContext) -> Result<ResponseBuilder, HandlerError> {
    Ok(ResponseBuilder::new().text("Hello, World!"))
}

async fn create_user_handler(mut req: RequestContext) -> Result<ResponseBuilder, HandlerError> {
    let user: User = req.json().await?;
    let created_user = create_user(user).await?;
    Ok(ResponseBuilder::new().status(StatusCode::Created).json(created_user))
}

// WebSocket example
async fn example_websocket() {
    let ws = WebSocketClient::new("ws://localhost:8080/ws")
        .connect()
        .await?;
    
    // Send a message
    ws.send_text("Hello WebSocket!").await?;
    
    // Receive messages
    while let Ok(message) = ws.receive().await {
        match message {
            Message::Text(text) => print(f"Received: {text}"),
            Message::Binary(data) => print(f"Received binary: {data.len()} bytes"),
            Message::Close(_) => break,
            _ => {}
        }
    }
}

// gRPC client example
async fn example_grpc_client() {
    let client = grpc::Client::new("http://localhost:50051")
        .with_timeout(Duration::seconds(10));
    
    let request = GetUserRequest { id: 123 };
    let response: GetUserResponse = client
        .call("GetUser", request)
        .await?;
    
    print(f"User: {response.user.name}");
}

// TCP server example
async fn example_tcp_server() {
    let listener = TcpListener::bind("127.0.0.1:8080".parse()?).await?;
    
    while let Ok((stream, addr)) = listener.accept().await {
        spawn async move {
            handle_client(stream, addr).await?;
        };
    }
}

async fn handle_client(mut stream: TcpStream, addr: SocketAddr) -> Result<(), SocketError> {
    let mut buffer = [0u8; 1024];
    
    while let Ok(n) = stream.read(&mut buffer).await {
        if n == 0 { break; }
        
        let message = str::from_utf8(&buffer[..n])?;
        print(f"Received from {addr}: {message}");
        
        let response = f"Echo: {message}";
        stream.write_all(response.as_bytes()).await?;
    }
    
    Ok(())
}
```

This comprehensive NetworkingLib provides:

1. **HTTP Support**: Full-featured client and server with middleware
2. **WebSocket Support**: Real-time bidirectional communication
3. **gRPC Support**: High-performance RPC with streaming
4. **Low-level Sockets**: TCP/UDP with full configuration options
5. **TLS/SSL**: Secure communication with certificate management
6. **DNS Resolution**: Comprehensive DNS query capabilities
7. **Connection Pooling**: Efficient connection reuse and management
8. **Error Handling**: Detailed error types for all networking operations
9. **Async-First**: Built for high-concurrency applications
10. **Production Ready**: Timeouts, retries, monitoring, and observability